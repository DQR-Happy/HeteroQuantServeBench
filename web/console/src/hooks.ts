import { useEffect, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api, terminal } from './api/client';
import type { Run } from './api/types';
import { followRun } from './streaming';

export function useRun(id?: string, submittedAt?: number) {
  const client = useQueryClient();
  const [connection, setConnection] = useState('snapshot');
  const [clientFirst, setClientFirst] = useState<number | null>(null);
  const query = useQuery({
    queryKey: ['run', id],
    queryFn: () => api<Run>(`/runs/${id}`),
    enabled: !!id,
    refetchInterval: (q) => (q.state.data && terminal(q.state.data.state) ? false : 2000),
  });
  useEffect(() => {
    if (!id) return;
    const abort = new AbortController();
    let cursor = 0;
    let measured = false;
    setClientFirst(null);
    void (async () => {
      while (!abort.signal.aborted) {
        try {
          setConnection('connected');
          cursor = await followRun(id, cursor, abort.signal, (event) => {
            cursor = event.seq;
            if (!measured && submittedAt && event.data.text) {
              measured = true;
              setClientFirst(performance.now() - submittedAt);
            }
            client.setQueryData<Run>(['run', id], (old) => {
              if (!old || old.seq >= event.seq) return old;
              if (event.kind === 'output.snapshot')
                return {
                  ...old,
                  seq: event.seq,
                  output: event.data.text ?? old.output,
                  metrics: event.data.metrics ?? old.metrics,
                };
              return old;
            });
          });
          const snapshot = await api<Run>(`/runs/${id}`);
          if (abort.signal.aborted) break;
          client.setQueryData(['run', id], snapshot);
          cursor = snapshot.seq;
          if (terminal(snapshot.state)) {
            setConnection('completed');
            break;
          }
        } catch {
          if (abort.signal.aborted) break;
          setConnection('reconnecting');
          try {
            const snapshot = await api<Run>(`/runs/${id}`);
            client.setQueryData(['run', id], snapshot);
            cursor = snapshot.seq;
            if (terminal(snapshot.state)) {
              setConnection('completed');
              break;
            }
          } catch {
            /* Keep observing the same ID; never create another request. */
          }
        }
        await new Promise<void>((resolve) => {
          const finish = () => {
            clearTimeout(timer);
            abort.signal.removeEventListener('abort', finish);
            resolve();
          };
          const timer = setTimeout(finish, 2000);
          abort.signal.addEventListener('abort', finish, { once: true });
        });
      }
    })();
    return () => abort.abort();
  }, [id, client, submittedAt]);
  return { ...query, connection, clientFirst };
}
