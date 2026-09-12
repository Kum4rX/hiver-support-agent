/**
 * Service abstraction layer.
 *
 * The shape below matches the FastAPI backend payload exactly, so switching
 * from the demo service to the real API is a single implementation swap.
 */

export type RiskLevel = "none" | "low" | "medium" | "high" | "critical";
export type AgentDecision = "auto_response" | "human_review" | "out_of_domain";

export interface RetrievedCase {
  similarity: number; // 0.0 - 1.0
  customer_text: string;
  support_text: string;
}

export interface Guardrails {
  length_ok: boolean;
  pii_safe: boolean;
  actionable: boolean;
}

export interface AnalysisResult {
  query: string;
  intent: string;
  secondary_intents: string[];
  confidence: number; // 0.0 - 1.0
  escalated: boolean;
  escalation_reason: string | null;
  risk_level: RiskLevel;
  decision: AgentDecision;
  routing_target: string;
  retrieved_cases: RetrievedCase[];
  response: string;
  response_length: number;
  guardrails: Guardrails;
  latency_ms: number;
  stages: PipelineStage[];
}

export interface PipelineStage {
  id: string;
  label: string;
  status: "complete" | "skipped" | "blocked";
  duration_ms: number;
  detail: string;
}

export interface EscalationRecord {
  id: string;
  customer: string;
  query: string;
  intent: string;
  risk_level: RiskLevel;
  escalation_reason: string;
  routing_target: string;
  created_at: string;
  status: "open" | "in_review" | "resolved";
}

export interface RecentRun {
  id: string;
  query: string;
  intent: string;
  decision: AgentDecision;
  risk_level: RiskLevel;
  confidence: number;
  latency_ms: number;
  created_at: string;
}

export interface KnowledgeDoc {
  id: string;
  similarity: number;
  customer_text: string;
  support_text: string;
  tags: string[];
}

export interface ExampleQuery {
  id: string;
  label: string;
  query: string;
  tone: "neutral" | "critical" | "warning" | "info";
}

/* ------------------------------------------------------------------ */
/* Demo data                                                           */
/* ------------------------------------------------------------------ */

export const EXAMPLE_QUERIES: ExampleQuery[] = [
  {
    id: "battery-normal",
    label: "Normal Battery Issue",
    tone: "neutral",
    query:
      "@AppleSupport my iPhone 13 battery drains from 100% to 30% in about three hours since the last update. Any fix?",
  },
  {
    id: "wifi",
    label: "Wi-Fi Connectivity Issue",
    tone: "neutral",
    query:
      "@AppleSupport my MacBook keeps dropping the Wi-Fi connection every few minutes while other devices stay connected.",
  },
  {
    id: "battery-critical",
    label: "Critical Battery Safety Hazard",
    tone: "critical",
    query:
      "@AppleSupport my iPhone battery is swelling and there is a burning smell coming from the back of the phone, it feels very hot.",
  },
  {
    id: "account",
    label: "Account Compromise",
    tone: "warning",
    query:
      "@AppleSupport someone logged into my Apple ID from another country and changed my password, I am locked out of my account.",
  },
  {
    id: "ood",
    label: "Windows/Dell Out-of-Domain",
    tone: "info",
    query:
      "@AppleSupport my Dell XPS running Windows 11 shows a blue screen every time I open Excel, how do I fix the driver?",
  },
  {
    id: "multi",
    label: "Multi-intent Issue",
    tone: "warning",
    query:
      "@AppleSupport after the iOS update my battery dies twice as fast and my phone will not stay connected to Wi-Fi either.",
  },
];

const CASES = {
  battery: [
    {
      similarity: 0.91,
      customer_text:
        "Battery percentage falls really fast after updating to the latest iOS, drops 40% in two hours.",
      support_text:
        "Check Settings > Battery > Battery Health & Charging for peak performance capability, review the app breakdown for high background usage, then restart the device. If capacity is below 80%, a service request is recommended.",
    },
    {
      similarity: 0.84,
      customer_text: "My phone battery lasts half a day now, was fine last week.",
      support_text:
        "Background App Refresh and location services are the usual causes after an update. Disable them for apps you rarely use and re-index for 24 hours before comparing usage.",
    },
    {
      similarity: 0.78,
      customer_text: "Battery drains while I sleep even in low power mode.",
      support_text:
        "Overnight drain is normally caused by a stuck sync process. Sign out of iCloud sync for Photos temporarily and check the Battery Usage graph in the morning.",
    },
  ],
  wifi: [
    {
      similarity: 0.89,
      customer_text: "MacBook keeps disconnecting from Wi-Fi while other devices are stable.",
      support_text:
        "Remove the network under System Settings > Wi-Fi > Advanced, then renew the DHCP lease and rejoin. If it persists, reset the network location and test on 5GHz only.",
    },
    {
      similarity: 0.82,
      customer_text: "Wi-Fi drops every few minutes after the latest macOS update.",
      support_text:
        "Delete the network preference files, restart in safe mode to check for third-party VPN interference, then rejoin the network.",
    },
    {
      similarity: 0.75,
      customer_text: "Wi-Fi says connected but no internet on my Mac.",
      support_text:
        "Renew the DHCP lease and set DNS manually. If other devices work, the router's band steering may be the cause.",
    },
  ],
  safety: [
    {
      similarity: 0.94,
      customer_text: "My battery looks puffy and the back panel is lifting away from the frame.",
      support_text:
        "Stop using and charging the device immediately, keep it away from flammable material, and book an in-person Genius Bar appointment. Do not attempt to remove the battery.",
    },
    {
      similarity: 0.9,
      customer_text: "Phone got extremely hot and smells like burning plastic.",
      support_text:
        "Power the device off, disconnect it from any charger, place it on a non-flammable surface and contact hardware support for an immediate safety inspection.",
    },
  ],
  account: [
    {
      similarity: 0.88,
      customer_text: "Someone accessed my Apple ID and changed the password without permission.",
      support_text:
        "Start account recovery from a trusted device, revoke unknown sessions and trusted devices, then enable two-factor authentication once access is restored.",
    },
    {
      similarity: 0.81,
      customer_text: "I got a login alert from a country I have never visited.",
      support_text:
        "Change the password immediately, review trusted devices, and check that the recovery email and phone number still belong to you.",
    },
  ],
};

export const KNOWLEDGE_CORPUS: KnowledgeDoc[] = [
  ...CASES.battery.map((c, i) => ({
    id: `kb-bat-${i}`,
    ...c,
    tags: ["battery", "power"],
  })),
  ...CASES.wifi.map((c, i) => ({ id: `kb-wifi-${i}`, ...c, tags: ["wifi", "network"] })),
  ...CASES.safety.map((c, i) => ({ id: `kb-safe-${i}`, ...c, tags: ["safety", "hardware"] })),
  ...CASES.account.map((c, i) => ({ id: `kb-acc-${i}`, ...c, tags: ["account", "security"] })),
  {
    id: "kb-update-0",
    similarity: 0.72,
    customer_text: "iOS update stuck on the progress bar for over an hour.",
    support_text:
      "Force restart the device, connect to a computer and use Finder to update without erasing content.",
    tags: ["software", "update"],
  },
  {
    id: "kb-store-0",
    similarity: 0.68,
    customer_text: "Charged twice for the same App Store subscription.",
    support_text:
      "Request a refund via reportaproblem.apple.com and share the order ID so billing can verify the duplicate charge.",
    tags: ["billing", "subscription"],
  },
];

export const RECENT_RUNS: RecentRun[] = [
  {
    id: "run-1042",
    query: "iPhone battery drains from 100% to 30% in three hours",
    intent: "BATTERY_POWER",
    decision: "auto_response",
    risk_level: "low",
    confidence: 0.94,
    latency_ms: 22,
    created_at: "2m ago",
  },
  {
    id: "run-1041",
    query: "Battery swelling and burning smell",
    intent: "BATTERY_POWER",
    decision: "human_review",
    risk_level: "critical",
    confidence: 0.97,
    latency_ms: 19,
    created_at: "8m ago",
  },
  {
    id: "run-1040",
    query: "MacBook keeps dropping Wi-Fi every few minutes",
    intent: "CONNECTIVITY",
    decision: "auto_response",
    risk_level: "low",
    confidence: 0.91,
    latency_ms: 25,
    created_at: "14m ago",
  },
  {
    id: "run-1039",
    query: "Dell XPS blue screen on Windows 11",
    intent: "OUT_OF_DOMAIN",
    decision: "out_of_domain",
    risk_level: "none",
    confidence: 0.88,
    latency_ms: 17,
    created_at: "21m ago",
  },
  {
    id: "run-1038",
    query: "Apple ID accessed from another country",
    intent: "ACCOUNT_ICLOUD",
    decision: "human_review",
    risk_level: "high",
    confidence: 0.93,
    latency_ms: 28,
    created_at: "33m ago",
  },
  {
    id: "run-1037",
    query: "Battery drain plus Wi-Fi drops after update",
    intent: "BATTERY_POWER",
    decision: "auto_response",
    risk_level: "medium",
    confidence: 0.79,
    latency_ms: 31,
    created_at: "48m ago",
  },
];

export const ESCALATIONS: EscalationRecord[] = [
  {
    id: "esc-2201",
    customer: "Demo Case #001",
    query: "My iPhone battery is swelling and there is a burning smell from the back.",
    intent: "BATTERY_POWER",
    risk_level: "critical",
    escalation_reason: "PHYSICAL_SAFETY_HAZARD",
    routing_target: "Hardware Safety Team",
    created_at: "Today 14:22",
    status: "in_review",
  },
  {
    id: "esc-2200",
    customer: "Demo Case #002",
    query: "Someone logged into my Apple ID from another country and changed my password.",
    intent: "ACCOUNT_ICLOUD",
    risk_level: "high",
    escalation_reason: "Account compromise — identity verification required",
    routing_target: "Account Security Team",
    created_at: "Today 13:05",
    status: "open",
  },
  {
    id: "esc-2199",
    customer: "Demo Case #003",
    query: "Phone got extremely hot while charging overnight and the screen is discoloured.",
    intent: "DEVICE_PERFORMANCE",
    risk_level: "critical",
    escalation_reason: "Thermal event reported during charging",
    routing_target: "Hardware Safety Team",
    created_at: "Today 11:47",
    status: "resolved",
  },
  {
    id: "esc-2198",
    customer: "Demo Case #004",
    query: "Charged three times for the same subscription and support chat closed on me.",
    intent: "PURCHASE_PAYMENT",
    risk_level: "medium",
    escalation_reason: "Repeat billing dispute — low retrieval confidence",
    routing_target: "Billing Specialists",
    created_at: "Yesterday 18:31",
    status: "open",
  },
  {
    id: "esc-2197",
    customer: "Demo Case #005",
    query: "Battery health dropped to 62% in four months, is this a defect?",
    intent: "BATTERY_POWER",
    risk_level: "medium",
    escalation_reason: "Possible warranty claim — human confirmation required",
    routing_target: "Warranty Review",
    created_at: "Yesterday 16:12",
    status: "resolved",
  },
];

/* ------------------------------------------------------------------ */
/* Demo analyzer                                                       */
/* ------------------------------------------------------------------ */

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function stages(opts: {
  retrieval?: "complete" | "skipped";
  generation?: "complete" | "blocked";
  intentDetail: string;
  riskDetail: string;
  retrievalDetail: string;
  generationDetail: string;
}): PipelineStage[] {
  return [
    {
      id: "preprocess",
      label: "Preprocessing",
      status: "complete",
      duration_ms: 1.4,
      detail: "Normalised handles, emoji and casing",
    },
    {
      id: "intent",
      label: "Intent Classification",
      status: "complete",
      duration_ms: 6.2,
      detail: opts.intentDetail,
    },
    {
      id: "risk",
      label: "Risk & Escalation",
      status: "complete",
      duration_ms: 2.1,
      detail: opts.riskDetail,
    },
    {
      id: "retrieval",
      label: "FAISS Retrieval",
      status: opts.retrieval ?? "complete",
      duration_ms: opts.retrieval === "skipped" ? 0 : 9.8,
      detail: opts.retrievalDetail,
    },
    {
      id: "generation",
      label: "Response Generation",
      status: "complete",
      duration_ms: 4.3,
      detail: opts.generationDetail,
    },
    {
      id: "guardrails",
      label: "Guardrails",
      status: "complete",
      duration_ms: 1.1,
      detail: "Length, PII and actionability checks passed",
    },
  ];
}

function build(partial: Omit<AnalysisResult, "response_length" | "latency_ms">): AnalysisResult {
  return {
    ...partial,
    response_length: partial.response.length,
    latency_ms: Math.round(
      partial.stages.reduce((sum, s) => sum + s.duration_ms, 0) * 10 + 8,
    ) / 10,
  };
}

function classify(q: string) {
  const t = q.toLowerCase();
  const safety = /(swell|swollen|burning|smoke|fire|melt|puff|explod|extremely hot|overheat)/.test(t);
  const battery = /(battery|charge|charging|drain|power)/.test(t);
  const wifi = /(wi-?fi|wifi|network|connect|bluetooth|internet)/.test(t);
  const account = /(apple id|account|password|hacked|compromis|locked out|login)/.test(t);
  const ood = /(windows|dell|hp |lenovo|android|samsung|excel|blue screen|driver)/.test(t);
  return { safety, battery, wifi, account, ood };
}

function analyzeSync(query: string): AnalysisResult {
  const f = classify(query);

  if (f.safety) {
    const response =
      "Please stop using and charging this device immediately and keep it away from flammable materials. Do not attempt to remove the battery. We are connecting you with our Hardware Safety Team now for an urgent inspection.";
    return build({
      query,
      intent: "BATTERY_POWER",
      secondary_intents: ["SECURITY"],
      confidence: 0.97,
      escalated: true,
      escalation_reason: "PHYSICAL_SAFETY_HAZARD",
      risk_level: "critical",
      decision: "human_review",
      routing_target: "Hardware Safety Team",
      retrieved_cases: CASES.safety,
      response,
      guardrails: { length_ok: response.length <= 280, pii_safe: true, actionable: true },
      stages: stages({
        intentDetail: "BATTERY_POWER (0.97) with SECURITY safety signal",
        riskDetail: "CRITICAL — deterministic safety override engaged",
        retrievalDetail: "Context retrieved; customer advice suppressed",
        generationDetail: "Deterministic safety response used",
      }),
    });
  }

  if (f.ood && !f.battery && !f.wifi && !f.account) {
    const response =
      "Thanks for reaching out. This looks like a Windows/Dell issue, which sits outside what this Apple support agent can help with. Your device manufacturer's support team will be best placed to resolve the driver error.";
    return build({
      query,
      intent: "OUT_OF_DOMAIN",
      secondary_intents: [],
      confidence: 0.88,
      escalated: false,
      escalation_reason: null,
      risk_level: "none",
      decision: "out_of_domain",
      routing_target: "No routing — boundary response returned",
      retrieved_cases: [],
      response,
      guardrails: { length_ok: response.length <= 280, pii_safe: true, actionable: true },
      stages: stages({
        retrieval: "skipped",
        intentDetail: "OUT_OF_DOMAIN (0.88) — non-Apple hardware referenced",
        riskDetail: "No risk signals",
        retrievalDetail: "Skipped — query outside supported corpus domain",
        generationDetail: "Polite boundary template returned",
      }),
    });
  }

  if (f.account) {
    const response =
      "Let's secure the account first: start account recovery from a trusted device, then revoke any unknown sessions and trusted devices. A security specialist is joining to verify your identity before we restore access.";
    return build({
      query,
      intent: "ACCOUNT_ICLOUD",
      secondary_intents: ["SECURITY"],
      confidence: 0.93,
      escalated: true,
      escalation_reason: "Account compromise — identity verification required",
      risk_level: "high",
      decision: "human_review",
      routing_target: "Account Security Team",
      retrieved_cases: CASES.account,
      response,
      guardrails: { length_ok: response.length <= 280, pii_safe: true, actionable: true },
      stages: stages({
        intentDetail: "ACCOUNT_ICLOUD (0.93)",
        riskDetail: "HIGH — unauthorised access reported",
        retrievalDetail: "2 account-security cases retrieved (mean sim 0.85)",
        generationDetail: "Grounded response with human handoff notice",
      }),
    });
  }

  if (f.battery && f.wifi) {
    const response =
      "Battery first: check Settings > Battery for high-usage apps and turn off Background App Refresh for them. Once drain settles, forget and rejoin your Wi-Fi network so the connection issue can be checked separately.";
    return build({
      query,
      intent: "BATTERY_POWER",
      secondary_intents: ["CONNECTIVITY"],
      confidence: 0.79,
      escalated: false,
      escalation_reason: null,
      risk_level: "medium",
      decision: "auto_response",
      routing_target: "Automated resolution — battery queue",
      retrieved_cases: [CASES.battery[0]!, CASES.wifi[0]!, CASES.battery[1]!],
      response,
      guardrails: { length_ok: response.length <= 280, pii_safe: true, actionable: true },
      stages: stages({
        intentDetail: "Primary BATTERY_POWER (0.79), secondary CONNECTIVITY (0.62)",
        riskDetail: "MEDIUM — multi-intent lowers single-intent confidence",
        retrievalDetail: "3 cases retrieved across both intents",
        generationDetail: "Primary intent leads, secondary context retained",
      }),
    });
  }

  if (f.wifi) {
    const response =
      "Try forgetting the network under Wi-Fi settings, then renew the DHCP lease and rejoin on the 5GHz band. If it still drops, restart the router and test in safe mode to rule out VPN or security software.";
    return build({
      query,
      intent: "CONNECTIVITY",
      secondary_intents: [],
      confidence: 0.91,
      escalated: false,
      escalation_reason: null,
      risk_level: "low",
      decision: "auto_response",
      routing_target: "Automated resolution — connectivity queue",
      retrieved_cases: CASES.wifi,
      response,
      guardrails: { length_ok: response.length <= 280, pii_safe: true, actionable: true },
      stages: stages({
        intentDetail: "CONNECTIVITY (0.91)",
        riskDetail: "LOW — routine troubleshooting",
        retrievalDetail: "3 cases retrieved (mean sim 0.82)",
        generationDetail: "Grounded in top-3 historical resolutions",
      }),
    });
  }

  if (f.battery) {
    const response =
      "Check Settings > Battery > Battery Health & Charging to see maximum capacity, then review which apps used most power today and turn off Background App Refresh for them. If capacity is under 80%, a battery service is recommended.";
    return build({
      query,
      intent: "BATTERY_POWER",
      secondary_intents: [],
      confidence: 0.94,
      escalated: false,
      escalation_reason: null,
      risk_level: "low",
      decision: "auto_response",
      routing_target: "Automated resolution — battery queue",
      retrieved_cases: CASES.battery,
      response,
      guardrails: { length_ok: response.length <= 280, pii_safe: true, actionable: true },
      stages: stages({
        intentDetail: "BATTERY_POWER (0.94)",
        riskDetail: "LOW — no safety keywords present",
        retrievalDetail: "3 cases retrieved (mean sim 0.84)",
        generationDetail: "Grounded in top-3 historical resolutions",
      }),
    });
  }

  const response =
    "Thanks for the details. Restart the device, make sure it is on the latest software version, and let us know exactly when the issue started. That will help narrow this down before we suggest next steps.";
  return build({
    query,
    intent: "HOW_TO_OTHER",
    secondary_intents: [],
    confidence: 0.61,
    escalated: false,
    escalation_reason: null,
    risk_level: "low",
    decision: "auto_response",
    routing_target: "Automated resolution — general queue",
    retrieved_cases: KNOWLEDGE_CORPUS.slice(9, 11).map((d) => ({
      similarity: d.similarity,
      customer_text: d.customer_text,
      support_text: d.support_text,
    })),
    response,
    guardrails: { length_ok: response.length <= 280, pii_safe: true, actionable: true },
    stages: stages({
      intentDetail: "HOW_TO_OTHER (0.61) — low confidence fallback",
      riskDetail: "LOW — no risk signals",
      retrievalDetail: "2 loosely related cases retrieved",
      generationDetail: "Clarifying response generated",
    }),
  });
}

const API_BASE_URL =
  (typeof import.meta !== "undefined" && import.meta.env && import.meta.env.VITE_API_BASE_URL) ||
  "http://localhost:8000";

/**
 * Analyze a single customer message using the real FastAPI Python AI Agent backend.
 * Falls back seamlessly to offline deterministic engine if the backend is unreachable.
 */
export async function analyzeCustomerMessage(message: string): Promise<AnalysisResult> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    if (res.ok) {
      const data = (await res.json()) as AnalysisResult;
      return data;
    }
    console.warn(`[Backend API] /api/analyze returned status ${res.status}`);
  } catch (err) {
    console.warn("[Backend API] Could not reach FastAPI server at " + API_BASE_URL + ", using offline mode.", err);
  }

  // Graceful offline fallback
  await sleep(200);
  return analyzeSync(message);
}

/** @deprecated use analyzeCustomerMessage */
export const analyzeTicket = analyzeCustomerMessage;

export async function searchKnowledgeBase(query: string): Promise<KnowledgeDoc[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, k: 6 }),
    });

    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data.results) && data.results.length > 0) {
        return data.results as KnowledgeDoc[];
      }
    }
  } catch (err) {
    console.warn("[Backend API] Could not reach FAISS search endpoint, using local corpus.", err);
  }

  await sleep(150);
  const terms = query.toLowerCase().split(/\s+/).filter((t) => t.length > 2);
  if (terms.length === 0) return KNOWLEDGE_CORPUS.slice(0, 5);
  return KNOWLEDGE_CORPUS.map((doc) => {
    const hay = `${doc.customer_text} ${doc.support_text} ${doc.tags.join(" ")}`.toLowerCase();
    const hits = terms.filter((t) => hay.includes(t)).length;
    return { ...doc, similarity: Math.min(0.97, 0.42 + hits * 0.16 + doc.similarity * 0.2) };
  })
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, 6);
}
