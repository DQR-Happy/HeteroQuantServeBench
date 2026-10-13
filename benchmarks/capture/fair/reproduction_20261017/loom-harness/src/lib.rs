// Unmodified official Loom Partitions library, with a bounded native ingress.
// Endpoint Q is sync/queryability, explicitly NOT a durable close endpoint.
use loom::{Partitions, raw_scan_u64};
use std::{ffi::{CStr,c_char},path::PathBuf,sync::{OnceLock,mpsc::{sync_channel,SyncSender}},
    collections::BTreeSet,fs::File,io::{BufWriter,Write},thread};
enum Command { Event(u32,Vec<u8>), Finish(SyncSender<i32>) }
static TX: OnceLock<SyncSender<Command>> = OnceLock::new();
fn ns() -> u64 { unsafe { let mut t=std::mem::zeroed(); libc::clock_gettime(libc::CLOCK_MONOTONIC,&mut t); t.tv_sec as u64*1_000_000_000+t.tv_nsec as u64 } }
fn field(b:&[u8])->u64 { u64::from_le_bytes(b[68..76].try_into().unwrap()) }
#[no_mangle]
pub unsafe extern "C" fn hf_open(path:*const c_char)->i32 {
    let path=PathBuf::from(CStr::from_ptr(path).to_str().unwrap());
    let profile=std::env::var("HQSB_FAIR_PROFILE").unwrap_or("1".into()).parse::<usize>().unwrap();
    let (tx,rx)=sync_channel([256,4096,65536][profile]);
    let (ready_tx,ready_rx)=sync_channel(1);
    TX.set(tx).ok().unwrap();
    thread::Builder::new().name("loom-owner".into()).stack_size(2*1024*1024).spawn(move||{
        let mut engine=Partitions::new(path.clone());
        for p in 0..16 { engine.define_source(p,0); engine.define_range_index_u64(p,0,field,0,10000,100); }
        let mut seen=BTreeSet::new();
        ready_tx.send(0).unwrap();
        while let Ok(command)=rx.recv() { match command {
            Command::Event(p,data)=>{engine.push(p as u64,0,&data);seen.insert(p);},
            Command::Finish(ack)=>{
                for &p in &seen { engine.sync(p as u64,0); }
                let q_end=ns();
                let query_start=ns(); let mut count=0u64;let mut reverse_errors=0u64;
                let reader=engine.reader();
                for &p in &seen {
                    let mut output=BufWriter::new(File::create(path.parent().unwrap().join(format!("loom-source-{p}.bin"))).unwrap());
                    let mut ctx=reader.indexed_query_ctx(&[(p as u64,0)]).unwrap();
                    let mut previous=u64::MAX;
                    raw_scan_u64(p as u64,0,0,u64::MAX,field,0,u64::MAX,|_,seq,data|{
                        // Warmups have run_id="warmup" and a separate sequence cohort.
                        if !data.starts_with(b"warmup\0") { if seq>=previous {reverse_errors+=1;} previous=seq; }
                        count+=1;output.write_all(&(data.len() as u32).to_le_bytes()).unwrap();output.write_all(data).unwrap();
                    },&mut ctx);
                    output.flush().unwrap();output.get_ref().sync_all().unwrap();
                }
                let query_end=ns();
                std::fs::write(path.parent().unwrap().join("loom-boundaries.json"),format!("{{\"q_end_ns\":{q_end},\"query_export_ns\":{},\"raw_scan_count\":{count},\"reverse_order_errors\":{reverse_errors},\"endpoint\":\"Q-sync; active blocks can be lost on process crash\"}}",query_end-query_start)).unwrap();
                ack.send(0).unwrap(); break;
            }
        }}
    }).unwrap();
    ready_rx.recv().unwrap()
}
#[no_mangle]
pub unsafe extern "C" fn hf_offer(run:*const c_char,p:u32,i:u64,ts:u64,body:*const c_char)->i32 {
    let run=CStr::from_ptr(run).to_bytes();let body=CStr::from_ptr(body).to_bytes();
    if run.len()>=64 || p>=16 || body.len()+88>=65500 {return -1;}
    let mut record=vec![0u8;64];record[..run.len()].copy_from_slice(run);
    record.extend_from_slice(&p.to_le_bytes());record.extend_from_slice(&i.to_le_bytes());
    record.extend_from_slice(&ts.to_le_bytes());record.extend_from_slice(&(body.len() as u32).to_le_bytes());
    record.extend_from_slice(body);
    if TX.get().unwrap().send(Command::Event(p,record)).is_ok(){0}else{-1}
}
#[no_mangle]
pub extern "C" fn hf_close()->i32 {
    let (tx,rx)=sync_channel(1);TX.get().unwrap().send(Command::Finish(tx)).unwrap();rx.recv().unwrap()
}
