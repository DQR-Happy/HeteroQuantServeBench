"""Dual-clock, bounded per-thread DMAT capture.

Adapted from the private Neutrino/ERETrace core_events.py supplied by the user;
the installed donor source is read-only. This HQSB extension adds actual thread
entry/global clocks, 48-byte events, classic cp.async and source-size semantics.
Input: one pruned PTX entry, one instruction/line, register +/- constant VAs.
All participating threads have bounded storage and exact attempted-event counts.
"""
import re
from neutrino.probe.cuda import COMMON_BUFFER_CALC


def instrument_memory_events(ptx: str, capacity: int):
    if not 1 <= capacity <= 2048:
        raise ValueError("capacity must be in [1, 2048]")
    if len(re.findall(r"\.entry\s", ptx)) != 1:
        raise ValueError("provide one pruned entry")
    code_only=re.sub(r'/\*.*?\*/|//[^\n]*','',ptx,flags=re.S)
    if re.search(r'(?<![\w$])(?:%?obs_[\w$]*|OBS_SKIP_[\w$]*|%buf\d+)\b',code_only):
        raise ValueError('input uses reserved event-instrumentation names')
    start=ptx.index('(',ptx.index('.entry'));end=ptx.index(')',start)
    params=ptx[start+1:end].strip().rstrip(',')
    ptx=ptx[:start+1]+'\n'+params+(',' if params else '')+'\n.param .u64 obs_records,\n.param .u64 obs_counts\n'+ptx[end:]
    body=ptx.index('{',ptx.index(')',start))
    entry='''.reg .u64 %obs_entry;
mov.u64 %obs_entry, %clock64;
'''+COMMON_BUFFER_CALC+f'''
.reg .u64 %obs_base, %obs_count_base, %obs_count, %obs_off;
ld.param.u64 %obs_base, [obs_records];
ld.param.u64 %obs_count_base, [obs_counts];
mul.wide.u32 %obs_off, %buf1, {capacity*48};
add.u64 %obs_base, %obs_base, %obs_off;
mul.wide.u32 %obs_off, %buf1, 8;
add.u64 %obs_count_base, %obs_count_base, %obs_off;
mov.u64 %obs_count, 0;
'''
    lines=ptx[body+1:].splitlines();out=[];sites=[]
    pattern=re.compile(r'^\s*(?P<p>@!?%[\w$]+\s+)?(?P<op>(?:ld|st)\.global[.\w:]*|cp\.async\.(?:ca|cg)\.shared\.global[.\w:]*)\s+(?P<args>[^;]+);')
    for line in lines:
        code=line.split('//',1)[0]
        terminal=re.match(r'^\s*(@!?%[\w$]+\s+)?(?:ret|exit)(?:\.uni)?;',code)
        if re.search(r'\b(?:ret|exit)(?:\.uni)?\s*;',code) and not terminal:
            raise ValueError('terminal instruction must occupy its own line')
        if terminal:
            out.append((terminal[1] or '')+'st.global.u64 [%obs_count_base], %obs_count;')
        m=pattern.match(code)
        if not m:
            if re.search(r'\b(?:ld|st)\.global\b|\bcp\.async\.(?:ca|cg)\.',code):
                raise ValueError('memory instruction must occupy its own line')
            out.append(line);continue
        if code[m.end():].strip():raise ValueError('multiple instructions on memory-event line')
        addresses=list(re.finditer(r'\[\s*(%[\w$]+)(?:\s*([+-])\s*(-?(?:0x[0-9a-fA-F]+|\d+)))?\s*\]',m['args']))
        asynchronous=m['op'].startswith('cp.async')
        address=addresses[-1] if addresses else None
        bits=re.search(r'\.([busf])(8|16|32|64)$',m['op'])
        if not address or len(addresses)!=(2 if asynchronous else 1) or (not asynchronous and not bits):
            raise ValueError(f'unsupported memory operand: {line}')
        source_bytes=None
        if asynchronous:
            tail=m['args'][address.end():].strip().lstrip(',').split(',')
            width=int(tail[0].strip())
            if width not in (4,8,16) or len(tail)>2: raise ValueError('unsupported cp.async operands')
            if len(tail)==2:
                source_bytes=tail[1].strip()
                if not re.fullmatch(r'%[\w$]+|\d+',source_bytes) or (source_bytes.isdigit() and not 0 <= int(source_bytes) <= width): raise ValueError('unsupported cp.async source size')
        else:
            vector=re.search(r'\.v([24])\.',m['op']);width=int(bits[2])//8*(int(vector[1]) if vector else 1)
        site=len(sites);sites.append(dict(site=site,opcode=m['op'],width=width,address=address[0], variable_width=source_bytes is not None))
        pred=(m['p'] or '').strip().lstrip('@')
        enabled=('not.pred %obs_enabled, '+pred[1:]+';' if pred.startswith('!') else 'mov.pred %obs_enabled, '+pred+';') if pred else 'setp.eq.u32 %obs_enabled, 0, 0;'
        addr_op=(('add' if address[2]=='+' else 'sub')+f'.u64 %obs_addr, {address[1]}, {address[3]};') if address[2] else f'mov.u64 %obs_addr, {address[1]};'
        out.append(f'''{{
.reg .pred %obs_enabled, %obs_space, %obs_nonzero;
.reg .u32 %obs_active, %obs_mask, %obs_site, %obs_width;
.reg .u64 %obs_addr, %obs_time, %obs_clock, %obs_index, %obs_dest;
{enabled}
mov.u32 %obs_width, {source_bytes or width};
setp.ne.u32 %obs_nonzero, %obs_width, 0;
and.pred %obs_enabled, %obs_enabled, %obs_nonzero;
activemask.b32 %obs_active;
vote.ballot.sync.b32 %obs_mask, %obs_enabled, %obs_active;
@!%obs_enabled bra OBS_SKIP_{site};
mov.u64 %obs_clock, %clock64;
mov.u64 %obs_time, %globaltimer;
{addr_op}
mov.u64 %obs_index, %obs_count;
add.u64 %obs_count, %obs_count, 1;
setp.lt.u64 %obs_space, %obs_index, {capacity};
@!%obs_space bra OBS_SKIP_{site};
mul.lo.u64 %obs_index, %obs_index, 48;
add.u64 %obs_dest, %obs_base, %obs_index;
mov.u32 %obs_site, {site};
st.global.v2.u64 [%obs_dest], {{%obs_time, %obs_clock}};
st.global.v2.u64 [%obs_dest+16], {{%obs_entry, %obs_addr}};
st.global.v4.u32 [%obs_dest+32], {{%obs_site, %buf1, %obs_mask, %obs_width}};
OBS_SKIP_{site}:
}}''')
        out.append(line)
    if not sites:
        raise ValueError('no supported memory sites')
    return ptx[:body+1]+'\n'+entry+'\n'+'\n'.join(out), sites
