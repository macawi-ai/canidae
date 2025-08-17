/* tslint:disable */
/* eslint-disable */
export function init_canidae(): void;
export function get_version(): string;
export class PackClient {
  free(): void;
  constructor(pack_id: string, member_id: string, ws_url: string);
  connect(): void;
  send_message(content: string): void;
  disconnect(): void;
  is_connected(): boolean;
  get_pack_id(): string;
  get_member_id(): string;
  set_message_callback(callback: Function): void;
  set_error_callback(callback: Function): void;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_packclient_free: (a: number, b: number) => void;
  readonly packclient_new: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
  readonly packclient_connect: (a: number) => [number, number];
  readonly packclient_send_message: (a: number, b: number, c: number) => [number, number];
  readonly packclient_disconnect: (a: number) => void;
  readonly packclient_is_connected: (a: number) => number;
  readonly packclient_get_pack_id: (a: number) => [number, number];
  readonly packclient_get_member_id: (a: number) => [number, number];
  readonly packclient_set_message_callback: (a: number, b: any) => void;
  readonly packclient_set_error_callback: (a: number, b: any) => void;
  readonly init_canidae: () => [number, number];
  readonly get_version: () => [number, number];
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_export_2: WebAssembly.Table;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_export_5: WebAssembly.Table;
  readonly __externref_table_dealloc: (a: number) => void;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly closure20_externref_shim: (a: number, b: number, c: any) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
