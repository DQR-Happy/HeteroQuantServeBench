#!/usr/bin/env sh
# Raise the VS Code Server / Extension Host V8 old-space ceiling.
#
# Why this exists (AGENTS.md §8.2/§8.3 #2, §8.4 02:55 复核):
#   The extension-host crash loop is "V8 heap limit reached -> SIGABRT ->
#   host auto-restart".  V8's default old-space limit is a *fixed* value
#   (~4 GB on 64-bit), NOT a fraction of physical RAM, so a 90 GB host does
#   not help by itself: the ceiling must be passed explicitly via NODE_OPTIONS.
#
#   The documented hook `~/.vscode-server/server-env-setup` is NEVER sourced
#   under the current "CLI + servers" layout (verified 2026-09-19: the probe
#   file was never created and `grep -a -c env-setup` is 0 in both the CLI
#   binary and `server-main.js`).  The launcher script that actually execs node
#   is `cli/servers/Stable-<commit>/server/bin/code-server`; it passes no heap
#   flag, so the extension host inherits none.  This script patches that
#   launcher, which is a *machine state* edit (not a tracked config), exactly
#   like AGENTS.md §8.3 describes.
#
# Properties:
#   * idempotent  — re-running is a no-op once the marker is present;
#   * reversible  — a `.orig` backup is kept next to the patched script;
#   * non-fatal   — a missing/unwritable launcher is reported, never forced;
#   * applies only after the SERVER restarts.  A window reload is NOT enough
#     (`Remote-SSH: Kill VS Code Server on Host`; closing the window keeps the
#     server alive, so the launcher is not re-read).
#
# Usage:
#   scripts/env/patch_vscode_server_heap.sh              # patch every version
#   scripts/env/patch_vscode_server_heap.sh --check      # report only
#   scripts/env/patch_vscode_server_heap.sh --revert     # restore .orig
#   VSCODE_HEAP_MB=12288 scripts/env/patch_vscode_server_heap.sh
#
# Verification after a server restart -- check the CMDLINE, not the environment:
#   ps -eo cmd | grep "[t]ype=extensionHost" | grep -o -- "--max-old-space-size=[0-9]*"
# (Do NOT verify with `/proc/<pid>/environ | grep NODE_OPTIONS`: `Hd()` in
#  server-main.js deletes NODE_OPTIONS from the forked extension host, so the
#  environment NEVER shows it even when the ceiling IS in effect.  That check
#  would report a false negative.)

set -eu

SERVERS_DIR="${HOME}/.vscode-server/cli/servers"
DEFAULT_HEAP_MB=8192
MODE="patch"

# Which ceiling to install?
#   * an explicit VSCODE_HEAP_MB always wins (that is how you deliberately change it);
#   * otherwise, if a launcher ALREADY carries a ceiling, keep it.
#     Without this rule, re-running the script with no argument would silently
#     *lower* an installed 16384 back to the 8192 default -- a silent downgrade,
#     and one that only shows up after the next server restart (2026-09-19: this
#     happened, hence the rule).
#   * otherwise fall back to the default.
if [ -n "${VSCODE_HEAP_MB:-}" ]; then
    HEAP_MB="$VSCODE_HEAP_MB"
    HEAP_SOURCE="VSCODE_HEAP_MB (explicit)"
else
    HEAP_MB=""
    for launcher in "$SERVERS_DIR"/Stable-*/server/bin/code-server; do
        [ -f "$launcher" ] || continue
        detected="$(grep 'server-main.js' "$launcher" 2>/dev/null | head -1 \
            | grep -o -- '--max-old-space-size=[0-9]*' | head -1 | sed 's/.*=//')"
        if [ -n "$detected" ]; then
            HEAP_MB="$detected"
            break
        fi
    done
    if [ -n "$HEAP_MB" ]; then
        HEAP_SOURCE="already installed on disk"
    else
        HEAP_MB="$DEFAULT_HEAP_MB"
        HEAP_SOURCE="default"
    fi
fi

MARKER="max-old-space-size=${HEAP_MB}"

for arg in "$@"; do
    case "$arg" in
        --check) MODE="check" ;;
        --revert) MODE="revert" ;;
        -h|--help) sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

if [ ! -d "$SERVERS_DIR" ]; then
    echo "server directory not found: $SERVERS_DIR"
    echo "（说明：当前没有 VS Code Server 布局；换机器/换实例后按本脚本重新应用）"
    exit 0
fi

found=0
patched=0
skipped=0
failed=0
installed=""   # ceiling actually present on disk (reported by --check)

# Every installed server version is patched, so a version bump does not silently
# lose the ceiling again.
for launcher in "$SERVERS_DIR"/Stable-*/server/bin/code-server; do
    [ -f "$launcher" ] || continue
    found=$((found + 1))

    case "$MODE" in
        check)
            # Report the ceiling that is ACTUALLY on disk, not the requested
            # default: printing `HEAP_MB` here would read "8192 MB" on a machine
            # whose launcher says 16384, i.e. a check that lies about the state.
            exec_line="$(grep 'server-main.js' "$launcher" 2>/dev/null | head -1)"
            detected="$(printf '%s\n' "$exec_line" | grep -o -- '--max-old-space-size=[0-9]*' | head -1 | sed 's/.*=//')"
            if [ -n "$detected" ]; then
                echo "[check] PATCHED, exec-line ceiling ${detected} MB: $launcher"
                installed="$detected"
            else
                echo "[check] NOT patched (no heap flag on the exec line): $launcher"
            fi
            continue
            ;;
        revert)
            backup="${launcher}.orig"
            if [ -f "$backup" ]; then
                cp "$backup" "$launcher"
                echo "[revert] restored $launcher"
                patched=$((patched + 1))
            else
                echo "[revert] no backup for $launcher"
                skipped=$((skipped + 1))
            fi
            continue
            ;;
    esac

    # Idempotency is judged on the *exec* line, not on the export: a launcher that
    # only has the export is exactly the half-fix that does not reach the host.
    exec_line="$(grep 'server-main.js' "$launcher" 2>/dev/null | head -1)"
    case "$exec_line" in
        *"$MARKER"*)
            echo "[patch] already patched (exec line), nothing to do: $launcher"
            skipped=$((skipped + 1))
            continue
            ;;
    esac
    if ! grep -q '^"\$ROOT/node"' "$launcher" 2>/dev/null; then
        echo "[patch] launcher layout changed, refusing to guess: $launcher" >&2
        failed=$((failed + 1))
        continue
    fi
    backup="${launcher}.orig"
    [ -f "$backup" ] || cp "$launcher" "$backup"
    # Two changes, because one is not enough (verified 2026-09-19):
    #
    #   (1) the exec line gets the flag as a REAL node CLI argument.  The
    #       extension host is forked with `execArgv` defaulting to the server's
    #       `process.execArgv` (server-main.js:
    #       `execArgv === void 0 && (e.execArgv = process.execArgv.filter(...))`),
    #       so a node CLI flag on the server is forwarded to the extension host.
    #   (2) `export NODE_OPTIONS=...` stays, for the server process itself and for
    #       any child that is not the extension host.
    #
    # NODE_OPTIONS alone does NOT reach the extension host: VS Code sanitises it
    # away (server-main.js: `Hd()` deletes DEBUG/NODE_OPTIONS/VSCODE_NODE_OPTIONS/
    # LD_PRELOAD/DYLD_INSERT_LIBRARIES from the forked env).  Both are applied so a
    # future VS Code change to either mechanism still leaves the ceiling in place.
    # `server-main.js` occurs exactly once in the pristine launcher (the exec
    # line), which the pre-flight check above already verified; matching on it
    # directly avoids fragile regex escaping of `$ROOT`.
    awk -v marker="--max-old-space-size=${HEAP_MB}" '
        /server-main\.js/ && !exec_done {
            printf "export NODE_OPTIONS=\"${NODE_OPTIONS:-} %s\"\n", marker
            printf "\"%s/node\" ${INSPECT:-} %s \"%s/out/server-main.js\" \"$@\"\n", "$ROOT", marker, "$ROOT"
            exec_done = 1
            next
        }
        { print }
    ' "$backup" > "$launcher"
    if grep -q -- "$MARKER" "$launcher" 2>/dev/null; then
        echo "[patch] patched: $launcher (backup: ${backup})"
        patched=$((patched + 1))
    else
        cp "$backup" "$launcher"
        echo "[patch] failed, restored original: $launcher" >&2
        failed=$((failed + 1))
    fi
done

echo "---"
echo "launchers found : $found"
echo "patched         : $patched"
echo "already ok/no-op: $skipped"
echo "failed          : $failed"
echo "heap ceiling    : ${HEAP_MB} MB (ceiling, not a preallocation; source: ${HEAP_SOURCE})"
[ -n "$installed" ] && echo "detected on disk: ${installed} MB"
echo ""
echo "改动天花板：VSCODE_HEAP_MB=<MB> 显式指定（不加则沿用磁盘上已有的值，不会静默升降）"
echo ""
echo "生效条件：必须重启 SERVER，窗口重载不够。"
echo "  客户端：Ctrl+Shift+P -> 'Remote-SSH: Kill VS Code Server on Host'"
echo "  等价： ssh <host> 'pkill -f .vscode-server'  （会断开当前连接，客户端会自动重连）"
echo "复验（查 cmdline，不要查 environ —— 见本脚本头部说明）："
echo "  ps -eo cmd | grep '[t]ype=extensionHost' | grep -o -- '--max-old-space-size=[0-9]*'"
exit 0
