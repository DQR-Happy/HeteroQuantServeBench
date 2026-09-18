export type Machine = {
  id: string;
  name: string;
  transport: 'local' | 'http' | 'ssh';
  base_url?: string;
  ssh_host?: string;
  ssh_user?: string;
  ssh_port?: number;
  remote_port?: number;
  identity_file?: string;
  token_env?: string;
};
export type Machines = { selected_id: string; control_node_id: string; items: Machine[] };
export async function activateMachine(id: string) {
  const { api } = await import('../../api/client');
  await api(`/machines/${encodeURIComponent(id)}/activate`, {});
  window.sessionStorage.setItem('hqsb.machine', id);
  // A new document discards queries, selected run IDs and model drafts together.
  window.location.assign('/devices');
}
