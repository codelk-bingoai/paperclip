import { randomUUID } from "node:crypto";
import express, { type IRouter } from "express";
import request from "supertest";
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import {
  agents,
  companies,
  createDb,
  projects,
  routineDocuments,
  routineRuns,
  routines,
  routineTriggers,
} from "@paperclipai/db";
import { getEmbeddedPostgresTestSupport, startEmbeddedPostgresTestDatabase } from "./helpers/embedded-postgres.js";

// CODA-38975: `routines.getDetail` hydrated its `assignee` with a bare
// `db.select().from(agents)`, so GET /api/routines/:id returned all 23 agent
// columns -- including `adapterConfig`, `runtimeConfig`, `permissions`, and
// the budget fields -- to any company member, regardless of grant. The fix
// (server/src/services/routines.ts) projects that hydration to the 5 columns
// `RoutineAgentSummary` declares. This suite is the regression guard: it runs
// against the real (unmocked) service and a real database, so it goes red
// against the pre-fix query and green once the projection lands -- a mock of
// `routineService` would bypass the fix's actual location (the SQL select)
// and could never go red for it.
//
// `access.js` is the one service mocked below. Every other route dependency
// (agents, routines, companies, projects) runs against the embedded Postgres
// database so the real queries execute.

const mockAccessService = vi.hoisted(() => ({
  decide: vi.fn(async ({ action }: { action?: string }) => ({
    allowed: action === "agent:read",
    reason: action === "agent:read" ? "allow_test_read" : "deny_missing_grant",
    explanation: action === "agent:read" ? "Allowed by test read grant." : "Missing test grant.",
  })),
  canUser: vi.fn(async () => false),
  getMembership: vi.fn(async () => null),
  listPrincipalGrants: vi.fn(async () => []),
  ensureMembership: vi.fn(),
  setPrincipalPermission: vi.fn(),
}));

vi.mock("../services/access.js", () => ({
  accessService: () => mockAccessService,
}));

const CANARY = "coda38975-canary-value";

// Every field the agents table carries that must never cross an embed
// boundary. Extend this list when a sensitive column is added, NOT the
// per-route assertions below.
const FORBIDDEN_IN_EMBED = [
  "adapterConfig",
  "runtimeConfig",
  "permissions",
  "budgetMonthlyCents",
  "spentMonthlyCents",
  "metadata",
] as const;

function findForbidden(node: unknown, path = "$"): string[] {
  const hits: string[] = [];
  if (Array.isArray(node)) {
    node.forEach((value, index) => hits.push(...findForbidden(value, `${path}[${index}]`)));
  } else if (node && typeof node === "object") {
    const obj = node as Record<string, unknown>;
    // Only an *embedded* agent is in scope: a direct read of the agent
    // routes themselves (GET /agents/:id, GET /companies/:id/agents, ...)
    // legitimately varies by grant and is excluded via ROUTE_ALLOW_LIST
    // below, not via this shape check.
    const isEmbeddedAgent =
      typeof obj.id === "string" && typeof obj.name === "string" && typeof obj.role === "string" && path !== "$";
    if (isEmbeddedAgent) {
      for (const field of FORBIDDEN_IN_EMBED) {
        if (field in obj) hits.push(`${path}.${field}`);
      }
    }
    for (const [key, value] of Object.entries(obj)) hits.push(...findForbidden(value, `${path}.${key}`));
  } else if (typeof node === "string" && node.includes(CANARY)) {
    hits.push(`${path} == CANARY`);
  }
  return hits;
}

// Routes whose response is a direct agent read, not an embed. These redact
// via a different, pre-existing mechanism (`redactForRestrictedAgentView` /
// `redactAgentConfiguration` in routes/agents.ts) that only blanks
// `adapterConfig`/`runtimeConfig` -- it does not strip `permissions` or the
// budget fields, and is out of scope for CODA-38975. Adding a route here is
// a reviewable decision; forgetting a per-route test is not.
const ROUTE_ALLOW_LIST = [
  (path: string) => path.startsWith("/agents"),
  (path: string) => path.startsWith("/companies/:companyId/agent"),
  (path: string) => path.startsWith("/companies/:companyId/org"),
  (path: string) => path === "/instance/scheduler-heartbeats",
];

function isAllowListed(path: string): boolean {
  return ROUTE_ALLOW_LIST.some((matcher) => matcher(path));
}

function listRegisteredGetRoutes(router: IRouter): string[] {
  const paths: string[] = [];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  for (const layer of (router as any).stack as any[]) {
    const route = layer.route;
    if (route && route.methods && route.methods.get) {
      paths.push(route.path as string);
    }
  }
  return paths;
}

function resolvePath(pattern: string, params: Record<string, string>): string {
  return pattern.replace(/:([A-Za-z0-9_]+)/g, (_match, name: string) => params[name] ?? "00000000-0000-4000-8000-000000000000");
}

const embeddedPostgresSupport = await getEmbeddedPostgresTestSupport();
const describeEmbeddedPostgres = embeddedPostgresSupport.supported ? describe.sequential : describe.skip;

if (!embeddedPostgresSupport.supported) {
  console.warn(
    `Skipping agent-embed redaction route tests on this host: ${embeddedPostgresSupport.reason ?? "unsupported environment"}`,
  );
}

describeEmbeddedPostgres("agent embed redaction across routes (CODA-38975)", () => {
  let db!: ReturnType<typeof createDb>;
  let tempDb: Awaited<ReturnType<typeof startEmbeddedPostgresTestDatabase>> | null = null;

  beforeAll(async () => {
    tempDb = await startEmbeddedPostgresTestDatabase("paperclip-embed-redaction-");
    db = createDb(tempDb.connectionString);
  }, 20_000);

  afterEach(async () => {
    await db.delete(routineTriggers);
    await db.delete(routineRuns);
    await db.delete(routineDocuments);
    await db.delete(routines);
    await db.delete(projects);
    await db.delete(agents);
    await db.delete(companies);
  });

  afterAll(async () => {
    await tempDb?.cleanup();
  });

  beforeEach(() => {
    mockAccessService.decide.mockImplementation(async ({ action }: { action?: string }) => ({
      allowed: action === "agent:read",
      reason: action === "agent:read" ? "allow_test_read" : "deny_missing_grant",
      explanation: action === "agent:read" ? "Allowed by test read grant." : "Missing test grant.",
    }));
  });

  async function createApp(actor: Record<string, unknown>) {
    const [{ errorHandler }, { routineRoutes }, { agentRoutes }] = await Promise.all([
      import("../middleware/index.js"),
      import("../routes/routines.js"),
      import("../routes/agents.js"),
    ]);
    const routineRouter = routineRoutes(db);
    const agentRouter = agentRoutes(db);
    const app = express();
    app.use(express.json());
    app.use((req, _res, next) => {
      (req as any).actor = actor;
      next();
    });
    app.use("/api", routineRouter);
    app.use("/api", agentRouter);
    app.use(errorHandler);
    return { app, routineRouter, agentRouter };
  }

  async function seedFixture() {
    const companyId = randomUUID();
    const callerAgentId = randomUUID();
    const otherAgentId = randomUUID();
    const projectId = randomUUID();
    const routineId = randomUUID();

    await db.insert(companies).values({
      id: companyId,
      name: "Paperclip",
      issuePrefix: `T${companyId.replace(/-/g, "").slice(0, 6).toUpperCase()}`,
      requireBoardApprovalForNewAgents: false,
    });

    await db.insert(agents).values({
      id: callerAgentId,
      companyId,
      name: "Caller",
      role: "engineer",
      status: "active",
      adapterType: "codex_local",
      adapterConfig: {},
      runtimeConfig: {},
      permissions: {},
    });

    await db.insert(agents).values({
      id: otherAgentId,
      companyId,
      name: "Other",
      role: "engineer",
      title: null,
      icon: null,
      status: "active",
      adapterType: "claude_local",
      adapterConfig: {
        env: { GITHUB_TOKEN: { type: "secret_ref", secretId: CANARY } },
      },
      runtimeConfig: { heartbeat: { enabled: true } },
      permissions: { canCreateAgents: true },
      budgetMonthlyCents: 5000,
      spentMonthlyCents: 1234,
      metadata: { note: CANARY },
    });

    await db.insert(projects).values({
      id: projectId,
      companyId,
      name: "Routine Project",
      status: "in_progress",
    });

    await db.insert(routines).values({
      id: routineId,
      companyId,
      projectId,
      title: "Daily routine",
      assigneeAgentId: otherAgentId,
    });

    return { companyId, callerAgentId, otherAgentId, projectId, routineId };
  }

  it("projects the routine assignee instead of returning the full agent row (negative control)", async () => {
    const { companyId, callerAgentId, otherAgentId, routineId } = await seedFixture();
    const { app } = await createApp({ type: "agent", agentId: callerAgentId, companyId });

    const res = await request(app).get(`/api/routines/${routineId}`);

    expect(res.status).toBe(200);
    expect(res.body.assignee).toEqual({
      id: otherAgentId,
      name: "Other",
      role: "engineer",
      title: null,
      icon: null,
    });
    expect(findForbidden(res.body)).toEqual([]);
  }, 20_000);

  it("sweeps every registered GET route on the mounted routers for a foreign agent embed", async () => {
    const { companyId, callerAgentId, otherAgentId, routineId } = await seedFixture();
    const { app, routineRouter, agentRouter } = await createApp({
      type: "agent",
      agentId: callerAgentId,
      companyId,
    });

    const checks: Array<{ path: string; status: number; forbidden: string[] }> = [];

    for (const pattern of listRegisteredGetRoutes(routineRouter)) {
      const path = resolvePath(pattern, { id: routineId, companyId, threadId: "none" });
      const res = await request(app).get(`/api${path}`);
      const forbidden = res.status === 200 && !isAllowListed(pattern) ? findForbidden(res.body) : [];
      checks.push({ path: pattern, status: res.status, forbidden });
    }

    for (const pattern of listRegisteredGetRoutes(agentRouter)) {
      const path = resolvePath(pattern, { id: otherAgentId, companyId });
      const res = await request(app).get(`/api${path}`);
      const forbidden = res.status === 200 && !isAllowListed(pattern) ? findForbidden(res.body) : [];
      checks.push({ path: pattern, status: res.status, forbidden });
    }

    const leaking = checks.filter((check) => check.forbidden.length > 0);
    expect(leaking).toEqual([]);
    // Sanity: the sweep actually exercised the vulnerable route and did not
    // silently no-op (e.g. because every path 404'd).
    expect(checks.some((check) => check.path === "/routines/:id" && check.status === 200)).toBe(true);
  }, 30_000);

  it("does not over-redact a caller holding the configuration-read grant (positive control)", async () => {
    const { companyId, callerAgentId, otherAgentId } = await seedFixture();
    mockAccessService.decide.mockImplementation(async ({ action }: { action?: string }) => ({
      allowed: action === "agent:read" || action === "agent_config:read",
      reason: "allow_test_grant",
      explanation: "Allowed by test grant.",
    }));
    const { app } = await createApp({ type: "agent", agentId: callerAgentId, companyId });

    const detailRes = await request(app).get(`/api/agents/${otherAgentId}`);
    const listRes = await request(app).get(`/api/companies/${companyId}/agents`);

    expect(detailRes.status).toBe(200);
    expect(detailRes.body.adapterConfig?.env?.GITHUB_TOKEN?.secretId).toBe(CANARY);
    expect(listRes.status).toBe(200);
    const otherInList = (listRes.body as Array<Record<string, unknown>>).find((a) => a.id === otherAgentId);
    expect((otherInList?.adapterConfig as Record<string, unknown> | undefined)?.env).toBeTruthy();
  }, 20_000);
});
