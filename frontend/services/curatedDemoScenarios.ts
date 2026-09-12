import type { ExampleQuery, EscalationRecord, CuratedScenario } from "./types";

/**
 * Curated evaluation & demonstration scenarios.
 * NOTE: These are curated scenarios for testing and demonstration, NOT live production customer traffic.
 */

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

export const CURATED_SCENARIOS: CuratedScenario[] = [
  {
    id: "scenario-1",
    label: "Normal Battery Drain",
    query: "iPhone 13 battery drains from 100% to 30% in three hours",
    intent: "BATTERY_POWER",
    expected_decision: "auto_response",
    expected_risk: "low",
    expected_routing: "Automated resolution — battery power queue",
    description: "Standard technical issue resolved via FAISS retrieval without escalation.",
  },
  {
    id: "scenario-2",
    label: "Wi-Fi Drops Post-Update",
    query: "MacBook keeps dropping Wi-Fi every few minutes while other devices are stable",
    intent: "CONNECTIVITY",
    expected_decision: "auto_response",
    expected_risk: "low",
    expected_routing: "Automated resolution — connectivity queue",
    description: "Network issue grounded in historical support resolutions.",
  },
  {
    id: "scenario-3",
    label: "Swollen Battery & Burning Smell",
    query: "Battery swelling and burning smell from phone",
    intent: "BATTERY_POWER",
    expected_decision: "human_review",
    expected_risk: "critical",
    expected_routing: "Hardware Safety Team",
    description: "Deterministic physical safety hazard triggering immediate emergency override.",
  },
  {
    id: "scenario-4",
    label: "Account Compromise",
    query: "Apple ID accessed from another country and password changed",
    intent: "SECURITY",
    expected_decision: "human_review",
    expected_risk: "high",
    expected_routing: "Account Security Specialist",
    description: "High-risk security breach routing to official iforgot.apple.com flow.",
  },
  {
    id: "scenario-5",
    label: "Windows 11 Dell Laptop BSOD",
    query: "Dell XPS blue screen on Windows 11 opening Excel",
    intent: "OUT_OF_DOMAIN",
    expected_decision: "out_of_domain",
    expected_risk: "none",
    expected_routing: "Boundary response — no FAISS hallucination",
    description: "Non-Apple hardware boundary detection redirecting to manufacturer support.",
  },
  {
    id: "scenario-6",
    label: "Dual Battery & Wi-Fi Issues",
    query: "Battery drain plus Wi-Fi drops after iOS update",
    intent: "BATTERY_POWER",
    expected_decision: "auto_response",
    expected_risk: "medium",
    expected_routing: "Automated resolution — dual-intent queue",
    description: "Multi-intent query addressing both issues within Twitter 280-char limit.",
  },
];

export const CURATED_ESCALATIONS: EscalationRecord[] = [
  {
    id: "esc-curated-001",
    customer: "Curated Safety Scenario #1",
    query: "My iPhone battery is swelling and there is a burning smell from the back.",
    intent: "BATTERY_POWER",
    risk_level: "critical",
    escalation_reason: "PHYSICAL_SAFETY_HAZARD",
    routing_target: "Hardware Safety Team",
    created_at: "Curated Benchmark Case",
    status: "in_review",
    source: "Curated safety scenario (Hazard recall benchmark)",
  },
  {
    id: "esc-curated-002",
    customer: "Curated Security Scenario #2",
    query: "Someone logged into my Apple ID from another country and changed my password.",
    intent: "ACCOUNT_ICLOUD",
    risk_level: "high",
    escalation_reason: "Account compromise — identity verification required",
    routing_target: "Account Security Specialist",
    created_at: "Curated Benchmark Case",
    status: "open",
    source: "Curated security scenario (Account protection benchmark)",
  },
  {
    id: "esc-curated-003",
    customer: "Curated Safety Scenario #3",
    query: "Phone got extremely hot while charging overnight and the screen is discoloured.",
    intent: "DEVICE_PERFORMANCE",
    risk_level: "critical",
    escalation_reason: "Thermal event reported during charging",
    routing_target: "Hardware Safety Team",
    created_at: "Curated Benchmark Case",
    status: "resolved",
    source: "Curated safety scenario (Hazard recall benchmark)",
  },
  {
    id: "esc-curated-004",
    customer: "Curated Billing Scenario #4",
    query: "Charged three times for the same subscription and support chat closed on me.",
    intent: "PURCHASE_PAYMENT",
    risk_level: "high",
    escalation_reason: "Financial fraud & repeat billing dispute",
    routing_target: "Payments & Billing Team",
    created_at: "Curated Benchmark Case",
    status: "open",
    source: "Curated financial dispute scenario",
  },
  {
    id: "esc-curated-005",
    customer: "Curated Hardware Scenario #5",
    query: "Battery health dropped to 62% in four months, is this a defect?",
    intent: "BATTERY_POWER",
    risk_level: "medium",
    escalation_reason: "Rapid battery degradation — warranty claim check required",
    routing_target: "Warranty Review Specialist",
    created_at: "Curated Benchmark Case",
    status: "resolved",
    source: "Curated warranty evaluation scenario",
  },
];
