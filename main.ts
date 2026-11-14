// free-model-aggregator-01: one server that fronts ten simulated free-tier providers with the
// OpenAI Chat Completions API and the Anthropic Messages API, in JSON and in server-sent events.
//
// How it works
//   The trial brief ($MAKEFUN_PROMPT_FILE) carries three JSON documents, found by their `schema`
//   member: the provider directory, the quality table (free models only, with ratings and
//   capabilities) and the workload manifest (each request's api, stream flag, weight and the
//   requirements a model must meet). At startup the server probes every provider's catalog route
//   in the provider's own wire, builds one catalog document from the listings plus the quality
//   table, listens on an ephemeral port, and registers {protocol_version, port, catalog} with the
//   env at $MAKEFUN_ENV_URL. `--list` prints that same catalog object and exits 0.
//
//   Each client request is routed to the best free model that meets the manifest's requirements
//   for it: candidates are ranked by the quality table's rating, providers are kept under a soft
//   call budget (the free-tier minimum each provider is known to grant, from the competition's
//   published provider table), and a provider that refuses three times in a trial is treated as
//   spent — a provider's own failure schedule holds at most two refusals, so a third refusal can
//   only mean its budget or its outage has begun, both permanent. A refusal fails over to the
//   next candidate inside the same request; a provider is never retried within one request,
//   because refusals are provider-level (budget, schedule, outage), never model-level.
//
//   Requests and answers cross wires through one canonical form: the client's system prompt,
//   messages (text, images, tool calls, tool results), tool definitions, tool choice, output cap,
//   stop sequences, temperature and JSON mode are extracted once, re-expressed in the chosen
//   provider's wire, and the provider's answer (text, usage, stop reason) is rendered back in the
//   client's wire, streamed or not as asked. A JSON-mode request from an OpenAI client is never
//   routed to an Anthropic-wire provider: that wire cannot carry response_format, so the provider
//   would echo the mode as absent.
//
//   Determinism: every choice derives from the brief, the requests and the providers' answers.
//   No wall clock (model listings and completions carry fixed timestamps), no randomness, one
//   request at a time, provider calls strictly sequential.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

type Json = Record<string, unknown>;
type Wire = 'openai' | 'anthropic' | 'gemini';
type ClientApi = 'openai' | 'anthropic';

const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const str = (v: unknown): string => (typeof v === 'string' ? v : '');
const numOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

/** Fixed timestamps for anything the server stamps itself: a moving clock would break replay. */
const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

// ---------------------------------------------------------------------------
// The brief

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

interface ProviderInfo {
  id: string;
  name: string;
  wire: Wire;
  auth: 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
  base: string;
  keyEnv: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating: number;
  context: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

interface Requires {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface ManifestEntry {
  id: string;
  api: ClientApi;
  stream: boolean;
  weight: number;
  requires: Requires;
}

interface Brief {
  providers: ProviderInfo[];
  ratings: RatingRow[];
  manifest: ManifestEntry[];
}

/** The index of the `}` that closes the object opening at `start`, skipping string contents. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in the brief text that carries a string `schema`, in order of appearance. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const end = objectEnd(text, at);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(at, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not a standalone JSON document: keep scanning.
    }
  }
  return out;
}

function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], manifest: [] };
  if (path === '') return empty;
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch {
    return empty;
  }
  const docs = documentsIn(text);
  const directory = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers));
  if (directory === undefined) return empty;
  const providers: ProviderInfo[] = [];
  for (const raw of directory.providers as unknown[]) {
    if (!isRecord(raw)) continue;
    const wire = raw.wire === 'openai' || raw.wire === 'anthropic' || raw.wire === 'gemini' ? raw.wire : null;
    const id = str(raw.id);
    if (wire === null || id === '') continue;
    const auth =
      raw.auth === 'bearer' || raw.auth === 'x-api-key' || raw.auth === 'x-goog-api-key' || raw.auth === 'none' ? raw.auth : 'bearer';
    const base = str(raw.path) !== '' ? str(raw.path) : str(raw.base_url) !== '' ? str(raw.base_url) : str(raw.base);
    providers.push({ id, name: str(raw.name) !== '' ? str(raw.name) : id, wire, auth, base, keyEnv: str(raw.key_env), key: str(raw.key) });
  }
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  const ratings: RatingRow[] = [];
  if (ratingsDoc !== undefined) {
    for (const raw of ratingsDoc.ratings as unknown[]) {
      if (!isRecord(raw) || str(raw.model) === '' || str(raw.provider) === '') continue;
      ratings.push({
        model: str(raw.model),
        provider: str(raw.provider),
        rating: numOrNull(raw.rating_milli) ?? 0,
        context: numOrNull(raw.context_length),
        tools: raw.tools === true,
        vision: raw.vision === true,
        json: raw.json === true,
      });
    }
  }
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  const manifest: ManifestEntry[] = [];
  if (workloadDoc !== undefined) {
    for (const raw of workloadDoc.manifest as unknown[]) {
      if (!isRecord(raw) || str(raw.id) === '') continue;
      const req = isRecord(raw.requires) ? raw.requires : {};
      manifest.push({
        id: str(raw.id),
        api: raw.api === 'anthropic' ? 'anthropic' : 'openai',
        stream: raw.stream === true,
        weight: raw.weight === 2 ? 2 : 1,
        requires: {
          ...(req.tools === true ? { tools: true } : {}),
          ...(req.vision === true ? { vision: true } : {}),
          ...(req.json === true ? { json: true } : {}),
          ...(numOrNull(req.min_context) !== null ? { min_context: numOrNull(req.min_context) as number } : {}),
        },
      });
    }
  }
  return { providers, ratings, manifest };
}

// ---------------------------------------------------------------------------
// Provider probing and the catalog

interface ProbedModel {
  id: string;
  displayName: string;
  free: boolean | null;
  context: number | null;
}

interface Probe {
  provider: ProviderInfo;
  ok: boolean;
  models: ProbedModel[];
}

/** Join a provider's base (a path like /p/groq, or an absolute URL) to the env's origin. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider's auth scheme asks for; the wire does not decide the header. */
function authHeaders(provider: ProviderInfo): Record<string, string> {
  if (provider.key === '' || provider.auth === 'none') return {};
  if (provider.auth === 'bearer') return { authorization: `Bearer ${provider.key}` };
  if (provider.auth === 'x-api-key') return { 'x-api-key': provider.key };
  if (provider.auth === 'x-goog-api-key') return { 'x-goog-api-key': provider.key };
  return {};
}

function listedModel(wire: Wire, raw: unknown): ProbedModel | null {
  if (!isRecord(raw)) return null;
  const id = wire === 'gemini' ? str(raw.name).replace(/^models\//, '') : str(raw.id);
  if (id === '') return null;
  return {
    id,
    displayName: str(raw.display_name) !== '' ? str(raw.display_name) : str(raw.displayName) !== '' ? str(raw.displayName) : id,
    free: typeof raw.free === 'boolean' ? raw.free : null,
    context: numOrNull(raw.context_length) ?? numOrNull(raw.context_window) ?? numOrNull(raw.inputTokenLimit),
  };
}

/** Ask one provider for its catalog, in its own wire; every failure leaves the provider out. */
async function probeProvider(envUrl: string, provider: ProviderInfo): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(5000) });
    if (!res.ok) return { provider, ok: false, models: [] };
    const body: unknown = await res.json();
    if (!isRecord(body)) return { provider, ok: false, models: [] };
    const list = provider.wire === 'gemini' ? body.models : body.data;
    if (!Array.isArray(list)) return { provider, ok: false, models: [] };
    const models: ProbedModel[] = [];
    for (const raw of list) {
      const model = listedModel(provider.wire, raw);
      if (model !== null) models.push(model);
    }
    return { provider, ok: true, models };
  } catch {
    return { provider, ok: false, models: [] };
  }
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface Catalog {
  providers: { id: string; base_url: string; key_env: string; wire: Wire; models: CatalogModel[] }[];
}

function ratingRow(brief: Brief, providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/**
 * The one catalog document: what `--list` prints and what the register body carries, byte for
 * byte. Its base_url is the provider's own path from the brief, never joined to the env's origin
 * (the origin differs between a live trial and its replay; the path names the provider in both).
 * A model's free flag is the quality table's where it has a row (the brief's table lists free
 * models only, so a row means free), else the listing's own word, else paid.
 */
function buildCatalog(brief: Brief, probes: Probe[]): Catalog {
  const providers: Catalog['providers'] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingRow(brief, probe.provider.id, model.id);
      models.push({ id: model.id, free: row !== null ? true : (model.free ?? false), context_length: row?.context ?? model.context });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.keyEnv, wire: probe.provider.wire, models });
  }
  return { providers };
}

// ---------------------------------------------------------------------------
// The canonical request: what a client asked, wire-neutral

type CanonBlock =
  | { kind: 'text'; text: string }
  | { kind: 'imageData'; mediaType: string; data: string }
  | { kind: 'imageUrl'; url: string }
  | { kind: 'toolUse'; id: string; name: string; input: unknown }
  | { kind: 'toolResult'; toolUseId: string; text: string };

interface CanonTool {
  name: string;
  description: string;
  schema: Json;
}

interface CanonRequest {
  system: string;
  messages: { role: 'user' | 'assistant'; blocks: CanonBlock[] }[];
  tools: CanonTool[];
  toolChoice: { mode: 'auto' | 'any' | 'none' | 'tool'; name: string } | null;
  maxTokens: number | null;
  stops: string[];
  temperature: number | null;
  topP: number | null;
  jsonMode: boolean;
  stream: boolean;
}

function parseDataUrl(url: string): { mediaType: string; data: string } | null {
  const match = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (match === null) return null;
  return { mediaType: match[1], data: match[2] };
}

/** The text of a message content: a string, or its text parts joined with newlines. */
function looseText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const part of content) {
    if (typeof part === 'string') parts.push(part);
    else if (isRecord(part) && typeof part.text === 'string') parts.push(part.text);
  }
  return parts.join('\n');
}

function openAIBlocks(content: unknown): CanonBlock[] {
  if (typeof content === 'string') return content === '' ? [] : [{ kind: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const blocks: CanonBlock[] = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') blocks.push({ kind: 'text', text: part.text });
    else if (part.type === 'image_url' && isRecord(part.image_url)) {
      const url = str(part.image_url.url);
      const data = parseDataUrl(url);
      if (data !== null) blocks.push({ kind: 'imageData', mediaType: data.mediaType, data: data.data });
      else if (url !== '') blocks.push({ kind: 'imageUrl', url });
    }
  }
  return blocks;
}

function anthropicBlocks(content: unknown): CanonBlock[] {
  if (typeof content === 'string') return content === '' ? [] : [{ kind: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const blocks: CanonBlock[] = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') blocks.push({ kind: 'text', text: part.text });
    else if (part.type === 'image' && isRecord(part.source)) {
      if (part.source.type === 'base64' && typeof part.source.data === 'string') {
        blocks.push({ kind: 'imageData', mediaType: str(part.source.media_type) !== '' ? str(part.source.media_type) : 'image/png', data: part.source.data });
      } else if (part.source.type === 'url' && str(part.source.url) !== '') {
        blocks.push({ kind: 'imageUrl', url: str(part.source.url) });
      }
    } else if (part.type === 'tool_use') {
      blocks.push({ kind: 'toolUse', id: str(part.id), name: str(part.name), input: part.input ?? {} });
    } else if (part.type === 'tool_result') {
      blocks.push({ kind: 'toolResult', toolUseId: str(part.tool_use_id), text: looseText(part.content) });
    }
  }
  return blocks;
}

function canonFromOpenAI(body: Json): CanonRequest {
  const systems: string[] = [];
  const messages: CanonRequest['messages'] = [];
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of rawMessages) {
    if (!isRecord(raw)) continue;
    const role = str(raw.role);
    if (role === 'system' || role === 'developer') {
      const text = looseText(raw.content);
      if (text !== '') systems.push(text);
    } else if (role === 'user') {
      messages.push({ role: 'user', blocks: openAIBlocks(raw.content) });
    } else if (role === 'assistant') {
      const blocks = openAIBlocks(raw.content);
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (isRecord(call) && isRecord(call.function)) {
            let input: unknown = {};
            try {
              input = JSON.parse(str(call.function.arguments));
            } catch {
              input = {};
            }
            blocks.push({ kind: 'toolUse', id: str(call.id), name: str(call.function.name), input });
          }
        }
      }
      messages.push({ role: 'assistant', blocks });
    } else if (role === 'tool') {
      messages.push({ role: 'user', blocks: [{ kind: 'toolResult', toolUseId: str(raw.tool_call_id), text: looseText(raw.content) }] });
    }
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (isRecord(raw) && isRecord(raw.function) && str(raw.function.name) !== '') {
        tools.push({
          name: str(raw.function.name),
          description: str(raw.function.description),
          schema: isRecord(raw.function.parameters) ? raw.function.parameters : {},
        });
      }
    }
  }
  let toolChoice: CanonRequest['toolChoice'] = null;
  if (body.tool_choice === 'auto') toolChoice = { mode: 'auto', name: '' };
  else if (body.tool_choice === 'required') toolChoice = { mode: 'any', name: '' };
  else if (body.tool_choice === 'none') toolChoice = { mode: 'none', name: '' };
  else if (isRecord(body.tool_choice) && isRecord(body.tool_choice.function)) {
    toolChoice = { mode: 'tool', name: str(body.tool_choice.function.name) };
  }
  const stops: string[] =
    typeof body.stop === 'string' ? [body.stop] : Array.isArray(body.stop) ? (body.stop as unknown[]).filter((s): s is string => typeof s === 'string') : [];
  const format = isRecord(body.response_format) ? body.response_format : null;
  return {
    system: systems.join('\n'),
    messages,
    tools,
    toolChoice,
    maxTokens: numOrNull(body.max_tokens) ?? numOrNull(body.max_completion_tokens),
    stops,
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    jsonMode: format !== null && (format.type === 'json_object' || format.type === 'json_schema'),
    stream: body.stream === true,
  };
}

function canonFromAnthropic(body: Json): CanonRequest {
  const messages: CanonRequest['messages'] = [];
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of rawMessages) {
    if (!isRecord(raw)) continue;
    const role = str(raw.role) === 'assistant' ? 'assistant' : 'user';
    messages.push({ role, blocks: anthropicBlocks(raw.content) });
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (isRecord(raw) && str(raw.name) !== '') {
        tools.push({ name: str(raw.name), description: str(raw.description), schema: isRecord(raw.input_schema) ? raw.input_schema : {} });
      }
    }
  }
  let toolChoice: CanonRequest['toolChoice'] = null;
  if (isRecord(body.tool_choice)) {
    if (body.tool_choice.type === 'auto') toolChoice = { mode: 'auto', name: '' };
    else if (body.tool_choice.type === 'any') toolChoice = { mode: 'any', name: '' };
    else if (body.tool_choice.type === 'none') toolChoice = { mode: 'none', name: '' };
    else if (body.tool_choice.type === 'tool') toolChoice = { mode: 'tool', name: str(body.tool_choice.name) };
  }
  return {
    system: looseText(body.system),
    messages,
    tools,
    toolChoice,
    maxTokens: numOrNull(body.max_tokens),
    stops: Array.isArray(body.stop_sequences) ? (body.stop_sequences as unknown[]).filter((s): s is string => typeof s === 'string') : [],
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    jsonMode: false,
    stream: body.stream === true,
  };
}

// ---------------------------------------------------------------------------
// Canonical request -> a provider's wire

function blocksText(blocks: CanonBlock[]): string {
  return blocks
    .filter((b): b is Extract<CanonBlock, { kind: 'text' }> => b.kind === 'text')
    .map((b) => b.text)
    .join('\n');
}

function toOpenAIRequest(canon: CanonRequest, model: string, stream: boolean): Json {
  const messages: Json[] = [];
  if (canon.system !== '') messages.push({ role: 'system', content: canon.system });
  for (const message of canon.messages) {
    if (message.role === 'user') {
      const parts: Json[] = [];
      for (const block of message.blocks) {
        if (block.kind === 'toolResult') {
          messages.push({ role: 'tool', tool_call_id: block.toolUseId, content: block.text });
        } else if (block.kind === 'text') {
          parts.push({ type: 'text', text: block.text });
        } else if (block.kind === 'imageData') {
          parts.push({ type: 'image_url', image_url: { url: `data:${block.mediaType};base64,${block.data}` } });
        } else if (block.kind === 'imageUrl') {
          parts.push({ type: 'image_url', image_url: { url: block.url } });
        }
      }
      if (parts.length === 1 && parts[0].type === 'text') messages.push({ role: 'user', content: parts[0].text });
      else if (parts.length > 0) messages.push({ role: 'user', content: parts });
    } else {
      const toolCalls: Json[] = [];
      for (const block of message.blocks) {
        if (block.kind === 'toolUse') {
          toolCalls.push({ id: block.id, type: 'function', function: { name: block.name, arguments: JSON.stringify(block.input ?? {}) } });
        }
      }
      const out: Json = { role: 'assistant', content: blocksText(message.blocks) };
      if (toolCalls.length > 0) out.tool_calls = toolCalls;
      messages.push(out);
    }
  }
  const body: Json = { model, messages, stream };
  if (canon.tools.length > 0) {
    body.tools = canon.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.schema } }));
  }
  if (canon.toolChoice !== null) {
    body.tool_choice =
      canon.toolChoice.mode === 'auto'
        ? 'auto'
        : canon.toolChoice.mode === 'any'
          ? 'required'
          : canon.toolChoice.mode === 'none'
            ? 'none'
            : { type: 'function', function: { name: canon.toolChoice.name } };
  }
  if (canon.maxTokens !== null) body.max_tokens = canon.maxTokens;
  if (canon.stops.length > 0) body.stop = canon.stops;
  if (canon.temperature !== null) body.temperature = canon.temperature;
  if (canon.topP !== null) body.top_p = canon.topP;
  if (canon.jsonMode) body.response_format = { type: 'json_object' };
  if (stream) body.stream_options = { include_usage: true };
  return body;
}

function toAnthropicRequest(canon: CanonRequest, model: string, stream: boolean): Json {
  const body: Json = { model, messages: [], stream };
  if (canon.system !== '') body.system = canon.system;
  const messages: Json[] = [];
  for (const message of canon.messages) {
    const content: Json[] = [];
    for (const block of message.blocks) {
      if (block.kind === 'text') content.push({ type: 'text', text: block.text });
      else if (block.kind === 'imageData') content.push({ type: 'image', source: { type: 'base64', media_type: block.mediaType, data: block.data } });
      else if (block.kind === 'imageUrl') content.push({ type: 'image', source: { type: 'url', url: block.url } });
      else if (block.kind === 'toolUse') content.push({ type: 'tool_use', id: block.id, name: block.name, input: block.input ?? {} });
      else if (block.kind === 'toolResult') content.push({ type: 'tool_result', tool_use_id: block.toolUseId, content: block.text });
    }
    messages.push({ role: message.role, content });
  }
  body.messages = messages;
  if (canon.tools.length > 0) {
    body.tools = canon.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.schema }));
  }
  if (canon.toolChoice !== null) {
    body.tool_choice =
      canon.toolChoice.mode === 'tool'
        ? { type: 'tool', name: canon.toolChoice.name }
        : { type: canon.toolChoice.mode === 'any' ? 'any' : canon.toolChoice.mode };
  }
  if (canon.maxTokens !== null) body.max_tokens = canon.maxTokens;
  if (canon.stops.length > 0) body.stop_sequences = canon.stops;
  if (canon.temperature !== null) body.temperature = canon.temperature;
  if (canon.topP !== null) body.top_p = canon.topP;
  return body;
}

function toGeminiRequest(canon: CanonRequest, stream: boolean): Json {
  const toolNameById = new Map<string, string>();
  for (const message of canon.messages) {
    for (const block of message.blocks) {
      if (block.kind === 'toolUse') toolNameById.set(block.id, block.name);
    }
  }
  const contents: Json[] = [];
  for (const message of canon.messages) {
    const parts: Json[] = [];
    for (const block of message.blocks) {
      if (block.kind === 'text') parts.push({ text: block.text });
      else if (block.kind === 'imageData') parts.push({ inlineData: { mimeType: block.mediaType, data: block.data } });
      else if (block.kind === 'imageUrl') parts.push({ fileData: { fileUri: block.url } });
      else if (block.kind === 'toolUse') parts.push({ functionCall: { name: block.name, args: isRecord(block.input) ? block.input : {} } });
      else if (block.kind === 'toolResult') {
        parts.push({ functionResponse: { name: toolNameById.get(block.toolUseId) ?? '', response: { result: block.text } } });
      }
    }
    contents.push({ role: message.role === 'assistant' ? 'model' : 'user', parts });
  }
  const body: Json = { contents };
  if (canon.system !== '') body.systemInstruction = { parts: [{ text: canon.system }] };
  if (canon.tools.length > 0) {
    body.tools = [{ functionDeclarations: canon.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.schema })) }];
  }
  const generationConfig: Json = {};
  if (canon.maxTokens !== null) generationConfig.maxOutputTokens = canon.maxTokens;
  if (canon.stops.length > 0) generationConfig.stopSequences = canon.stops;
  if (canon.temperature !== null) generationConfig.temperature = canon.temperature;
  if (canon.topP !== null) generationConfig.topP = canon.topP;
  if (canon.jsonMode) generationConfig.responseMimeType = 'application/json';
  if (Object.keys(generationConfig).length > 0) body.generationConfig = generationConfig;
  void stream; // Gemini chooses streaming by URL action, not by body.
  return body;
}

function providerCall(provider: ProviderInfo, envUrl: string, model: string, canon: CanonRequest): { url: string; headers: Record<string, string>; body: Json } {
  const base = joinBase(envUrl, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', accept: canon.stream ? 'text/event-stream, application/json' : 'application/json', ...authHeaders(provider) };
  if (provider.wire === 'openai') {
    return { url: `${base}/v1/chat/completions`, headers, body: toOpenAIRequest(canon, model, canon.stream) };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    return { url: `${base}/v1/messages`, headers, body: toAnthropicRequest(canon, model, canon.stream) };
  }
  const action = canon.stream ? 'streamGenerateContent' : 'generateContent';
  return { url: `${base}/v1beta/models/${encodeURIComponent(model)}:${action}`, headers, body: toGeminiRequest(canon, canon.stream) };
}

// ---------------------------------------------------------------------------
// A provider's answer -> canonical

interface CanonAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stop: string;
}

interface SseFrame {
  event: string;
  data: string;
}

function sseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  for (const block of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (block.trim() === '') continue;
    let event = 'message';
    const data: string[] = [];
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).trim());
    }
    frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function geminiText(candidate: unknown): string {
  if (!isRecord(candidate) || !isRecord(candidate.content) || !Array.isArray(candidate.content.parts)) return '';
  return (candidate.content.parts as unknown[])
    .map((part) => (isRecord(part) && typeof part.text === 'string' ? part.text : ''))
    .join('');
}

function parseUpstreamAnswer(wire: Wire, rawBody: string, stream: boolean): CanonAnswer | null {
  try {
    if (stream) {
      if (wire === 'openai') {
        let text = '';
        let input = 0;
        let output = 0;
        let stop = 'stop';
        let sawUsage = false;
        for (const frame of sseFrames(rawBody)) {
          if (frame.data === '' || frame.data === '[DONE]') continue;
          const chunk = JSON.parse(frame.data) as Json;
          if (isRecord(chunk.usage)) {
            sawUsage = true;
            input = numOrNull(chunk.usage.prompt_tokens) ?? input;
            output = numOrNull(chunk.usage.completion_tokens) ?? output;
          }
          const choice = Array.isArray(chunk.choices) ? chunk.choices[0] : undefined;
          if (isRecord(choice)) {
            if (isRecord(choice.delta) && typeof choice.delta.content === 'string') text += choice.delta.content;
            if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
          }
        }
        return sawUsage || text !== '' ? { text, inputTokens: input, outputTokens: output, stop } : null;
      }
      if (wire === 'anthropic') {
        let text = '';
        let input = 0;
        let output = 0;
        let stop = 'end_turn';
        for (const frame of sseFrames(rawBody)) {
          if (frame.data === '') continue;
          const data = JSON.parse(frame.data) as Json;
          if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
            input = numOrNull(data.message.usage.input_tokens) ?? input;
          } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta') {
            text += str(data.delta.text);
          } else if (frame.event === 'message_delta') {
            if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
            if (isRecord(data.usage)) output = numOrNull(data.usage.output_tokens) ?? output;
          }
        }
        return { text, inputTokens: input, outputTokens: output, stop };
      }
      let text = '';
      let input = 0;
      let output = 0;
      let stop = 'STOP';
      for (const frame of sseFrames(rawBody)) {
        if (frame.data === '') continue;
        const data = JSON.parse(frame.data) as Json;
        const candidate = Array.isArray(data.candidates) ? data.candidates[0] : undefined;
        text += geminiText(candidate);
        if (isRecord(candidate) && typeof candidate.finishReason === 'string') stop = candidate.finishReason;
        if (isRecord(data.usageMetadata)) {
          input = numOrNull(data.usageMetadata.promptTokenCount) ?? input;
          output = numOrNull(data.usageMetadata.candidatesTokenCount) ?? output;
        }
      }
      return { text, inputTokens: input, outputTokens: output, stop };
    }
    const body = JSON.parse(rawBody) as Json;
    if (wire === 'openai') {
      const choice = Array.isArray(body.choices) ? body.choices[0] : undefined;
      if (!isRecord(choice) || !isRecord(choice.message)) return null;
      const usage = isRecord(body.usage) ? body.usage : {};
      return {
        text: looseText(choice.message.content),
        inputTokens: numOrNull(usage.prompt_tokens) ?? 0,
        outputTokens: numOrNull(usage.completion_tokens) ?? 0,
        stop: str(choice.finish_reason) !== '' ? str(choice.finish_reason) : 'stop',
      };
    }
    if (wire === 'anthropic') {
      const usage = isRecord(body.usage) ? body.usage : {};
      return {
        text: looseText(body.content),
        inputTokens: numOrNull(usage.input_tokens) ?? 0,
        outputTokens: numOrNull(usage.output_tokens) ?? 0,
        stop: str(body.stop_reason) !== '' ? str(body.stop_reason) : 'end_turn',
      };
    }
    const candidate = Array.isArray(body.candidates) ? body.candidates[0] : undefined;
    const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
    return {
      text: geminiText(candidate),
      inputTokens: numOrNull(usage.promptTokenCount) ?? 0,
      outputTokens: numOrNull(usage.candidatesTokenCount) ?? 0,
      stop: isRecord(candidate) && str(candidate.finishReason) !== '' ? str(candidate.finishReason) : 'STOP',
    };
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Canonical answer -> the client's wire

/** Map any provider's stop spelling onto the client's wire, by family. */
function stopFor(stop: string, wire: ClientApi): string {
  const family =
    stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS'
      ? 'length'
      : stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use'
        ? 'tools'
        : stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY'
          ? 'filter'
          : 'end';
  if (wire === 'openai') return family === 'length' ? 'length' : family === 'tools' ? 'tool_calls' : family === 'filter' ? 'content_filter' : 'stop';
  return family === 'length' ? 'max_tokens' : family === 'tools' ? 'tool_use' : family === 'filter' ? 'refusal' : 'end_turn';
}

function renderOpenAI(answer: CanonAnswer, model: string, requestId: string, stream: boolean): string {
  const stop = stopFor(answer.stop, 'openai');
  if (!stream) {
    return JSON.stringify({
      id: `chatcmpl-${requestId}`,
      object: 'chat.completion',
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: stop, logprobs: null }],
      usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
    });
  }
  const chunk = (delta: Json, finish: string | null, withUsage = false): string => {
    const base: Json = { id: `chatcmpl-${requestId}`, object: 'chat.completion.chunk', created: FIXED_CREATED, model };
    if (withUsage) {
      base.choices = [];
      base.usage = { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens };
    } else {
      base.choices = [{ index: 0, delta, finish_reason: finish, logprobs: null }];
    }
    return `data: ${JSON.stringify(base)}\n\n`;
  };
  return (
    chunk({ role: 'assistant', content: '' }, null) +
    chunk({ content: answer.text }, null) +
    chunk({}, stop) +
    chunk({}, null, true) +
    'data: [DONE]\n\n'
  );
}

function renderAnthropic(answer: CanonAnswer, model: string, requestId: string, stream: boolean): string {
  const stop = stopFor(answer.stop, 'anthropic');
  if (!stream) {
    return JSON.stringify({
      id: `msg_${requestId}`,
      type: 'message',
      role: 'assistant',
      model,
      content: [{ type: 'text', text: answer.text }],
      stop_reason: stop,
      stop_sequence: null,
      usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
    });
  }
  const frame = (event: string, data: Json): string => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  return (
    frame('message_start', {
      type: 'message_start',
      message: {
        id: `msg_${requestId}`,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }) +
    frame('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }) +
    frame('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: answer.text } }) +
    frame('content_block_stop', { type: 'content_block_stop', index: 0 }) +
    frame('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: answer.outputTokens } }) +
    frame('message_stop', { type: 'message_stop' })
  );
}

function clientError(api: ClientApi, status: number, message: string): { status: number; payload: string } {
  if (api === 'anthropic') {
    const type = status === 400 ? 'invalid_request_error' : 'api_error';
    return { status, payload: JSON.stringify({ type: 'error', error: { type, message } }) };
  }
  const type = status === 400 ? 'invalid_request_error' : 'api_error';
  return { status, payload: JSON.stringify({ error: { message, type, param: null, code: null } }) };
}

// ---------------------------------------------------------------------------
// Routing

/**
 * The fewest calls each provider's free tier is known to grant (the competition's published
 * provider table gives a range; the low end is the safe plan). Candidates under their cap are
 * tried before candidates over it; the cap steers allocation, it never blocks a request.
 */
const SOFT_CALL_CAP: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

/** Refusals after which a provider is certainly spent: its own schedule holds at most two. */
const DEAD_AFTER_REFUSALS = 3;

interface ProviderState {
  counted: number; // calls the provider itself counts: successes and spend-refusals
  refusals: number; // spend-refusals only (402 / 429 / 5xx / unreachable)
  dead: boolean;
}

interface Candidate {
  provider: ProviderInfo;
  model: string;
  rating: number;
  tier: number;
}

function meets(requires: Requires, row: RatingRow): boolean {
  if (requires.tools === true && !row.tools) return false;
  if (requires.vision === true && !row.vision) return false;
  if (requires.json === true && !row.json) return false;
  if (typeof requires.min_context === 'number' && (row.context === null || row.context < requires.min_context)) return false;
  return true;
}

/** Requirements derived from the request body itself, when the manifest has no row for it. */
function bodyRequirements(api: ClientApi, body: Json): Requires {
  const requires: Requires = {};
  const messages = Array.isArray(body.messages) ? body.messages : [];
  if (Array.isArray(body.tools) && body.tools.length > 0) requires.tools = true;
  for (const raw of messages) {
    if (!isRecord(raw) || !Array.isArray(raw.content)) continue;
    for (const part of raw.content as unknown[]) {
      if (!isRecord(part)) continue;
      if (api === 'openai' && part.type === 'image_url') requires.vision = true;
      if (api === 'anthropic' && part.type === 'image') requires.vision = true;
    }
  }
  if (api === 'openai' && isRecord(body.response_format) && body.response_format.type === 'json_object') requires.json = true;
  return requires;
}

class Router {
  states: Map<string, ProviderState> = new Map();
  brief: Brief;
  probes: Probe[];
  envUrl: string;

  constructor(brief: Brief, probes: Probe[], envUrl: string) {
    this.brief = brief;
    this.probes = probes;
    this.envUrl = envUrl;
  }

  state(providerId: string): ProviderState {
    let found = this.states.get(providerId);
    if (found === undefined) {
      found = { counted: 0, refusals: 0, dead: false };
      this.states.set(providerId, found);
    }
    return found;
  }

  /** Ranked candidates for one request: live-and-under-cap first, then over-cap, then spent. */
  candidates(requires: Requires, canon: CanonRequest, api: ClientApi): Candidate[] {
    const out: Candidate[] = [];
    this.probes.forEach((probe, providerIndex) => {
      if (!probe.ok) return;
      const state = this.state(probe.provider.id);
      const cap = SOFT_CALL_CAP[probe.provider.id] ?? Number.POSITIVE_INFINITY;
      for (const model of probe.models) {
        const row = ratingRow(this.brief, probe.provider.id, model.id);
        if (row === null) continue; // No row in the free table: not a free model, never a pick.
        if (!meets(requires, row)) continue;
        if (api === 'openai' && canon.jsonMode && probe.provider.wire === 'anthropic') continue;
        const tier = state.dead ? 2 : state.counted >= cap ? 1 : 0;
        out.push({ provider: probe.provider, model: model.id, rating: row.rating, tier });
        void providerIndex;
      }
    });
    const order = new Map(this.brief.providers.map((p, i) => [p.id, i]));
    out.sort(
      (a, b) =>
        a.tier - b.tier ||
        b.rating - a.rating ||
        (order.get(a.provider.id) ?? 0) - (order.get(b.provider.id) ?? 0) ||
        (a.model < b.model ? -1 : a.model > b.model ? 1 : 0),
    );
    return out;
  }

  /**
   * Serve one client request. Returns the HTTP status and payload for the client's wire.
   * Provider calls are sequential: one success ends the request, one refusal fails over.
   */
  async serve(api: ClientApi, body: Json, entry: ManifestEntry | undefined): Promise<{ status: number; payload: string; stream: boolean }> {
    const canon = api === 'openai' ? canonFromOpenAI(body) : canonFromAnthropic(body);
    const requires = entry !== undefined ? entry.requires : bodyRequirements(api, body);
    const requestId = entry?.id ?? 'request';
    const tried = new Set<string>();
    for (const candidate of this.candidates(requires, canon, api)) {
      if (tried.has(candidate.provider.id)) continue;
      tried.add(candidate.provider.id);
      const state = this.state(candidate.provider.id);
      const call = providerCall(candidate.provider, this.envUrl, candidate.model, canon);
      let status = 0;
      let raw = '';
      try {
        const res = await fetch(call.url, { method: 'POST', headers: call.headers, body: JSON.stringify(call.body), signal: AbortSignal.timeout(25000) });
        status = res.status;
        raw = await res.text();
      } catch {
        status = 0;
      }
      if (status >= 200 && status < 300) {
        state.counted += 1;
        const answer = parseUpstreamAnswer(candidate.provider.wire, raw, canon.stream);
        if (answer !== null) {
          const payload = api === 'openai' ? renderOpenAI(answer, candidate.model, requestId, canon.stream) : renderAnthropic(answer, candidate.model, requestId, canon.stream);
          return { status: 200, payload, stream: canon.stream };
        }
        continue; // A 2xx we cannot read: not a refusal, just try the next candidate.
      }
      if (status === 402 || status === 429 || status >= 500 || status === 0) {
        if (status !== 0) state.counted += 1;
        state.refusals += 1;
        if (state.refusals >= DEAD_AFTER_REFUSALS) state.dead = true;
      }
      // Any other non-2xx (a malformed-pick 400, a wrong-key 401): the provider did not spend,
      // and neither did its budget; move on to the next candidate.
    }
    const failure = clientError(api, 502, 'no free provider could serve this request');
    return { status: failure.status, payload: failure.payload, stream: false };
  }
}

// ---------------------------------------------------------------------------
// The HTTP surface

function freeModelIds(catalog: Catalog): string[] {
  const ids: string[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (model.free && !ids.includes(model.id)) ids.push(model.id);
    }
  }
  return ids;
}

function modelsListing(api: ClientApi, catalog: Catalog): string {
  const ids = freeModelIds(catalog);
  if (api === 'anthropic') {
    return JSON.stringify({
      data: ids.map((id) => ({ type: 'model', id, display_name: id, created_at: FIXED_CREATED_AT })),
      has_more: false,
      first_id: ids[0] ?? null,
      last_id: ids[ids.length - 1] ?? null,
    });
  }
  return JSON.stringify({
    object: 'list',
    data: ids.map((id) => ({ id, object: 'model', created: FIXED_CREATED, owned_by: 'free-model-aggregator' })),
  });
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > 4_194_304) {
        reject(new Error('request body too large'));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

function sendJson(res: ServerResponse, status: number, payload: string): void {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(payload);
}

function sendStream(res: ServerResponse, payload: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(payload);
}

// ---------------------------------------------------------------------------
// Boot: probe, maybe --list, listen, register, serve until the env says done.

const envUrl = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const brief = readBrief(process.env.MAKEFUN_PROMPT_FILE ?? '');
const probes: Probe[] =
  envUrl === ''
    ? brief.providers.map((provider) => ({ provider, ok: false, models: [] }))
    : await Promise.all(brief.providers.map((provider) => probeProvider(envUrl, provider)));
const catalog = buildCatalog(brief, probes);

if (process.argv.includes('--list')) {
  process.stdout.write(`${JSON.stringify(catalog)}\n`);
  process.exit(0);
}

const router = new Router(brief, probes, envUrl);
let manifestIndex = 0;
let queue: Promise<unknown> = Promise.resolve();
const enqueue = <T>(fn: () => Promise<T>): Promise<T> => {
  const run = queue.then(fn, fn);
  queue = run.catch(() => undefined);
  return run;
};

const server = createServer((req, res) => {
  void (async () => {
    const path = (req.url ?? '').split('?')[0];
    try {
      if (req.method === 'GET' && path === '/v1/models') {
        const api: ClientApi = req.headers['x-api-key'] !== undefined ? 'anthropic' : 'openai';
        sendJson(res, 200, modelsListing(api, catalog));
        return;
      }
      if (req.method === 'POST' && (path === '/v1/chat/completions' || path === '/v1/messages')) {
        const api: ClientApi = path === '/v1/messages' ? 'anthropic' : 'openai';
        const raw = await readBody(req);
        let body: Json;
        try {
          const parsed: unknown = JSON.parse(raw);
          if (!isRecord(parsed)) throw new Error('body is not an object');
          body = parsed;
        } catch {
          const failure = clientError(api, 400, 'request body is not valid JSON');
          sendJson(res, failure.status, failure.payload);
          return;
        }
        const result = await enqueue(() => {
          const entry = brief.manifest[manifestIndex];
          manifestIndex += 1;
          return router.serve(api, body, entry);
        });
        if (result.stream) sendStream(res, result.payload);
        else sendJson(res, result.status, result.payload);
        return;
      }
      sendJson(res, 404, JSON.stringify({ error: { message: 'not found', type: 'invalid_request_error', param: null, code: null } }));
    } catch {
      if (!res.headersSent) sendJson(res, 500, JSON.stringify({ error: { message: 'internal error', type: 'api_error', param: null, code: null } }));
      else res.end();
    }
  })();
});

await new Promise<void>((resolve) => server.listen(0, '0.0.0.0', resolve));
const address = server.address();
const port = typeof address === 'object' && address !== null ? address.port : 0;

try {
  const res = await fetch(`${envUrl}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: 1, port, catalog }),
    signal: AbortSignal.timeout(10000),
  });
  if (!res.ok) throw new Error(`register answered ${res.status}`);
} catch (error) {
  console.error(`free-model-aggregator: registration failed: ${error instanceof Error ? error.message : String(error)}`);
  process.exit(1);
}

const poll = setInterval(() => {
  void (async () => {
    try {
      const res = await fetch(`${envUrl}/work`, { signal: AbortSignal.timeout(5000) });
      if (!res.ok) return;
      const body: unknown = await res.json();
      if (isRecord(body) && body.done === true) {
        clearInterval(poll);
        server.close();
        process.exit(0);
      }
    } catch {
      // The env answers when it answers; keep polling.
    }
  })();
}, 100);

process.on('SIGTERM', () => process.exit(0));
process.on('SIGINT', () => process.exit(0));
