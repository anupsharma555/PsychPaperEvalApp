import { useEffect, useMemo, useRef, useState } from "react";
import {
  API_BASE,
  DesktopApiError,
  cleanOrphans,
  exportDiagnosticsBundle,
  fetchBootstrap,
  fetchDocumentMedia,
  fetchJobs,
  fetchReport,
  fetchReportSaveStatus,
  fetchReportSummary,
  fetchRuntimeEvents,
  fetchStatus,
  getDesktopRuntimeInfo,
  openLogsFolder,
  pauseProcessing,
  recoverProcessing,
  restartBackend,
  resumeProcessing,
  saveReport,
  submitFromUpload,
  submitFromUrl,
} from "./api";

const WORKFLOW_STEPS = [
  { id: "select_source", label: "Select Source" },
  { id: "validate", label: "Validate Input" },
  { id: "submitted", label: "Submit Job" },
  { id: "monitoring", label: "Monitor" },
  { id: "review", label: "Review Report" },
];

const SUPPLEMENT_EXTS = [
  ".zip",
  ".csv",
  ".tsv",
  ".xlsx",
  ".xls",
  ".pdf",
  ".png",
  ".jpg",
  ".jpeg",
  ".tif",
  ".tiff",
  ".txt",
];

const EASTERN_TZ = "America/New_York";
const ET_DATE_TIME_FORMATTER = new Intl.DateTimeFormat("en-US", {
  timeZone: EASTERN_TZ,
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "numeric",
  minute: "2-digit",
  second: "2-digit",
  hour12: true,
  timeZoneName: "short",
});

const SUPPLEMENT_MARKER_RE =
  /\b(supplement(?:ary|al)?|suppl|appendix|extended data|supporting (?:information|info|data)|figure\s*s\d+|fig(?:ure)?\s*s\d+|table\s*s\d+|s\d+\s*(?:fig(?:ure)?|table))\b/i;

function nowLabel() {
  return ET_DATE_TIME_FORMATTER.format(new Date());
}

function normalizeDoi(input) {
  const match = String(input || "").trim().match(/10\.\d{4,9}\/\S+/i);
  return match ? match[0] : "";
}

function resolveUrlAndDoiInputs(urlInput, doiInput) {
  const rawUrl = String(urlInput || "").trim();
  const normalizedDoi = normalizeDoi(doiInput) || normalizeDoi(rawUrl);
  let resolvedUrl = rawUrl;
  if (!resolvedUrl && normalizedDoi) {
    resolvedUrl = `https://doi.org/${normalizedDoi}`;
  }
  if (resolvedUrl && !/^https?:\/\//i.test(resolvedUrl)) {
    const doiCandidate = normalizeDoi(resolvedUrl);
    if (doiCandidate) {
      resolvedUrl = `https://doi.org/${doiCandidate}`;
    }
  }
  return { resolvedUrl, normalizedDoi };
}

function normalizeErrorSentence(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function composeUiErrorMessage(message, nextAction) {
  const base = String(message || "").trim();
  const action = String(nextAction || "").trim();
  if (!base) return action;
  if (!action) return base;
  const baseNorm = normalizeErrorSentence(base);
  const actionNorm = normalizeErrorSentence(action);
  if (!baseNorm || !actionNorm) return `${base} ${action}`.trim();
  if (baseNorm.includes(actionNorm) || actionNorm.includes(baseNorm)) {
    return base.length >= action.length ? base : action;
  }
  return `${base} ${action}`.trim();
}

function toUiErrorMessage(error) {
  if (error instanceof DesktopApiError) {
    return composeUiErrorMessage(error.message, error.nextAction);
  }
  return String(error?.message || error || "Unknown error");
}

function sourceKindLabel(kind) {
  if (kind === "url") return "URL / DOI";
  if (kind === "upload") return "Upload";
  return String(kind || "unknown");
}

function formatEtDateTime(value) {
  if (!value) return "-";
  const date = value instanceof Date ? value : parseBackendDate(value);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return ET_DATE_TIME_FORMATTER.format(date);
}

function parseBackendDate(value) {
  const raw = String(value || "").trim();
  if (!raw) return new Date(NaN);
  const date = new Date(raw);
  if (!Number.isNaN(date.getTime())) return date;
  const normalized = raw.replace(" ", "T");
  if (!/[zZ]|[+-]\d{2}:\d{2}$/.test(normalized)) {
    const utcDate = new Date(`${normalized}Z`);
    if (!Number.isNaN(utcDate.getTime())) return utcDate;
  }
  return new Date(normalized);
}

function formatEvent(event) {
  return event?.message || event?.kind || "event";
}

function asArray(value) {
  if (Array.isArray(value)) return value;
  if (value === null || value === undefined) return [];
  return [value];
}

function normalizeEvidenceRefs(item) {
  const refs = [];
  const seen = new Set();
  const pushRef = (value) => {
    const ref = String(value || "").trim();
    if (!ref) return;
    if (ref.toLowerCase() === "unknown") return;
    const key = ref.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    refs.push(ref);
  };
  if (item && typeof item === "object") {
    pushRef(item.anchor);
    asArray(item.evidence_refs || item.evidence || item.refs).forEach((value) => pushRef(value));
  }
  return refs;
}

function formatEvidenceRef(ref) {
  const value = String(ref || "").trim();
  if (!value) return "";
  const lower = value.toLowerCase();
  const figureTokenMatch = value.match(/(?:^|[^a-z0-9])(?:fig(?:ure)?|f)\s*[_:-]?\s*(\d+[a-z]?)/i);
  const tableTokenMatch = value.match(/(?:^|[^a-z0-9])table\s*[_:-]?\s*(\d+[a-z]?)/i);
  const suppFigureTokenMatch = value.match(/supp(?:lement(?:ary)?)?[^a-z0-9]{0,6}(?:fig(?:ure)?|f)\s*[_:-]?\s*(\d+[a-z]?)/i);
  const suppTableTokenMatch = value.match(/supp(?:lement(?:ary)?)?[^a-z0-9]{0,6}table\s*[_:-]?\s*(\d+[a-z]?)/i);
  if (suppFigureTokenMatch) return `Supplement Figure ${suppFigureTokenMatch[1].toUpperCase()}`;
  if (suppTableTokenMatch) return `Supplement Table ${suppTableTokenMatch[1].toUpperCase()}`;
  if (figureTokenMatch) return `Figure ${figureTokenMatch[1].toUpperCase()}`;
  if (tableTokenMatch) return `Table ${tableTokenMatch[1].toUpperCase()}`;
  const plainFigure = value.match(/^(?:fig(?:ure)?[_:\s-]*)?f?(\d+[a-z]?)$/i);
  if (plainFigure && /^f/i.test(value)) {
    return `Figure ${plainFigure[1].toUpperCase()}`;
  }
  const plainTable = value.match(/^(?:table[_:\s-]*)?t?(\d+[a-z]?)$/i);
  if (plainTable && /^t/i.test(value)) {
    return `Table ${plainTable[1].toUpperCase()}`;
  }
  const suppFigure = value.match(/^s(?:upp(?:lement)?)?[_:\s-]*f(?:ig(?:ure)?)?[_:\s-]*(\d+[a-z]?)$/i);
  if (suppFigure) {
    return `Supplement Figure ${suppFigure[1].toUpperCase()}`;
  }
  const suppTable = value.match(/^s(?:upp(?:lement)?)?[_:\s-]*t(?:able)?[_:\s-]*(\d+[a-z]?)$/i);
  if (suppTable) {
    return `Supplement Table ${suppTable[1].toUpperCase()}`;
  }
  if (lower.startsWith("figure:")) {
    const token = value.split(":").slice(1).join(":");
    const pageMatch = token.match(/^page[_\s-]*(\d+)/i);
    if (pageMatch) return `Figure (page ${pageMatch[1]})`;
    const suppMatch = token.match(/supp(?:lement(?:ary)?)?[^a-z0-9]{0,6}(?:fig(?:ure)?|f)\s*[_:-]?\s*(\d+[a-z]?)/i);
    if (suppMatch) return `Supplement Figure ${suppMatch[1].toUpperCase()}`;
    const figMatch = token.match(/(?:fig(?:ure)?|f)\s*[_:-]?\s*(\d+[a-z]?)/i);
    if (figMatch) return `Figure ${figMatch[1].toUpperCase()}`;
    return "Figure";
  }
  if (lower.startsWith("table:")) {
    const token = value.split(":").slice(1).join(":");
    const pageMatch = token.match(/^page[_\s-]*(\d+)/i);
    if (pageMatch) return `Table (page ${pageMatch[1]})`;
    const suppMatch = token.match(/supp(?:lement(?:ary)?)?[^a-z0-9]{0,6}table\s*[_:-]?\s*(\d+[a-z]?)/i);
    if (suppMatch) return `Supplement Table ${suppMatch[1].toUpperCase()}`;
    const tableMatch = token.match(/table\s*[_:-]?\s*(\d+[a-z]?)/i);
    if (tableMatch) return `Table ${tableMatch[1].toUpperCase()}`;
    return "Table";
  }
  if (lower.startsWith("section:")) {
    const parts = value.split(":").slice(1).filter(Boolean);
    return parts.length > 0 ? `Section ${parts.join(" > ")}` : value;
  }
  if (lower.startsWith("supp")) {
    return `Supplement ${value.replace(/^supp(?:lement)?[:_\s-]*/i, "").trim() || value}`;
  }
  return value;
}

function evidenceContextFromItem(item) {
  const refs = normalizeEvidenceRefs(item);
  const primaryRef = refs[0] || "";
  const primaryLabel = formatEvidenceRef(primaryRef);
  const secondaryLabels = refs
    .slice(1, 4)
    .map((ref) => formatEvidenceRef(ref))
    .filter((label) => label && label !== primaryLabel);
  const findingId = String(item?.finding_id || item?.id || "").trim();
  return { primaryRef, primaryLabel, secondaryLabels, findingId };
}

function contextualizeStatement(statement, primaryLabel) {
  const text = String(statement || "").trim();
  const label = String(primaryLabel || "").trim();
  if (!text || !label) return text;
  const lower = label.toLowerCase();
  let out = text;
  if (lower.startsWith("figure")) {
    out = out.replace(/^figure\s+[a-z0-9._-]+(?:\s+[a-z0-9._-]+){0,5}\s+shows\s+/i, "shows ");
    out = out.replace(/^figure\s+[a-z0-9._-]+(?:\s+[a-z0-9._-]+){0,5}\s+/i, "");
    out = out.replace(/\b[Tt]he figure\b/g, label);
    out = out.replace(/\b[Ff]igure\b(?!\s*[a-z]?\d)/g, label);
    if (!out.toLowerCase().startsWith(label.toLowerCase())) {
      out = `${label}: ${out}`;
    }
  } else if (lower.startsWith("table")) {
    out = out.replace(/^table\s+[a-z0-9._-]+(?:\s+[a-z0-9._-]+){0,5}\s+shows\s+/i, "shows ");
    out = out.replace(/^table\s+[a-z0-9._-]+(?:\s+[a-z0-9._-]+){0,5}\s+/i, "");
    out = out.replace(/\b[Tt]he table\b/g, label);
    out = out.replace(/\b[Tt]able\b(?!\s*[a-z]?\d)/g, label);
    if (!out.toLowerCase().startsWith(label.toLowerCase())) {
      out = `${label}: ${out}`;
    }
  } else if (lower.startsWith("supplement")) {
    out = out.replace(/\b[Tt]he supplementary (?:figure|table|material)\b/g, label);
    out = out.replace(/\b[Tt]he supplement(?:ary)?\b/g, label);
  }
  return out;
}

function normalizeInlineRefLabels(text) {
  let out = String(text || "");
  out = out.replace(/\bsource:\s*([A-Za-z0-9:_-]+)/gi, (_, token) => {
    const label = formatEvidenceRef(token);
    return `source: ${label || token}`;
  });
  out = out.replace(/\brefs:\s*([A-Za-z0-9:,_\-\s]+)/gi, (_, rawRefs) => {
    const tokens = String(rawRefs || "")
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
    if (tokens.length === 0) return "refs:";
    const labels = tokens.map((token) => formatEvidenceRef(token) || token);
    return `refs: ${labels.join(", ")}`;
  });
  return out;
}

function toDisplayLine(item) {
  if (item === null || item === undefined) return "";
  if (typeof item === "string" || typeof item === "number" || typeof item === "boolean") {
    const raw = stripConfidenceTag(String(item));
    const matches = [...raw.matchAll(/\[([^\]]+)\]/g)];
    if (matches.length === 0) return normalizeInlineRefLabels(raw);
    const ref = String(matches[matches.length - 1][1] || "").trim();
    const label = formatEvidenceRef(ref);
    return normalizeInlineRefLabels(contextualizeStatement(raw, label));
  }
  if (typeof item === "object") {
    const statement = stripConfidenceTag(
      item.statement ||
        item.summary ||
        item.claim ||
        item.finding ||
        item.result ||
        item.text ||
        item.title
    );
    const { primaryRef, primaryLabel } = evidenceContextFromItem(item);
    const contextualStatement = contextualizeStatement(statement, primaryLabel);
    if (statement) {
      return `${contextualStatement}${primaryRef ? ` [${primaryRef}]` : ""}`;
    }
    try {
      return JSON.stringify(item);
    } catch {
      return String(item);
    }
  }
  return String(item);
}

function assetLabel(item, fallback) {
  const base = String(item?.caption || item?.anchor || fallback || "").trim() || String(fallback || "");
  if (base.length <= 180) return base;
  return `${base.slice(0, 177)}...`;
}

function mediaAssetHref(item) {
  // Keep original remote URL precedence for classification/deduping.
  const value = String(item?.asset_url || item?.source_proxy_url || item?.image_url || "").trim();
  return value;
}

function isTableLikeMediaItem(item) {
  const anchor = String(item?.anchor || "").trim().toLowerCase();
  const caption = String(item?.caption || "").trim().toLowerCase();
  const figureId = String(item?.figure_id || "").trim().toLowerCase();
  if (/^(t\d+[a-z]?)$/.test(anchor)) return true;
  if (/^table[:_\s-]*\d+/i.test(anchor)) return true;
  if (/^table\s+\d+/i.test(caption)) return true;
  if (/^table\b/i.test(caption)) return true;
  if (/^t\d+[a-z]?$/i.test(figureId)) return true;
  return false;
}

function isSupplementMediaItem(item) {
  const kind = String(item?.asset_kind || "").toLowerCase();
  if (kind === "supp") return true;
  const href = mediaAssetHref(item).toLowerCase();
  const rawAssetHref = String(item?.asset_url || "").toLowerCase();
  const proxyHref = String(item?.source_proxy_url || "").toLowerCase();
  const caption = String(item?.caption || "").toLowerCase();
  const anchor = String(item?.anchor || "").toLowerCase();
  const figureId = String(item?.figure_id || "").toLowerCase();
  const tableId = String(item?.table_id || "").toLowerCase();
  const markerBlob = `${caption} ${anchor} ${figureId} ${tableId} ${href} ${rawAssetHref} ${proxyHref}`;
  if (markerBlob.includes("/doi/suppl/") || markerBlob.includes("suppl_file")) return true;
  if (caption.includes("supplement") || anchor.includes("supplement") || anchor.includes("suppl")) return true;
  if (SUPPLEMENT_MARKER_RE.test(markerBlob)) return true;
  return false;
}

function isImageHref(value) {
  const href = String(value || "").trim().toLowerCase().split("?", 1)[0];
  if (/\/api\/documents\/\d+\/media\/\d+\/image$/i.test(href)) return true;
  return href.endsWith(".png") || href.endsWith(".jpg") || href.endsWith(".jpeg") || href.endsWith(".gif") || href.endsWith(".webp") || href.endsWith(".svg") || href.endsWith(".tif") || href.endsWith(".tiff");
}

function isPdfHref(value) {
  const href = String(value || "").trim().toLowerCase();
  return href.endsWith(".pdf") || href.includes(".pdf?");
}

function mediaViewerKind(value) {
  if (isImageHref(value)) return "image";
  if (isPdfHref(value)) return "document";
  return "document";
}

function dedupeMediaItems(items) {
  const out = [];
  const seen = new Set();
  for (const item of asArray(items)) {
    if (!item || typeof item !== "object") continue;
    const hrefKey = mediaAssetHref(item);
    const textKey = `${String(item.anchor || "").trim()}|${String(item.caption || "").trim()}`;
    const key = String(hrefKey || textKey || item.chunk_id || "").trim().toLowerCase();
    if (!key || seen.has(key)) continue;
    seen.add(key);
    out.push(item);
  }
  return out;
}

function stripMethodsLabel(value) {
  return String(value || "").replace(/^\s*\[[a-z0-9_\- ]+\]\s*/i, "").trim();
}

const CONFIDENCE_MARKER_RE =
  /\s*(?:\((?:model\s+)?confidence[:\s]*\d{1,3}(?:\.\d+)?%?\)|(?:model\s+)?confidence[:\s]+\d{1,3}(?:\.\d+)?%|\(\d{1,3}(?:\.\d+)?%\))\s*/gi;
const CANONICAL_STATEMENT_TOKEN_RE = /[a-z0-9]+(?:\.[0-9]+)?/g;

function stripConfidenceTag(value) {
  return String(value || "")
    .replace(CONFIDENCE_MARKER_RE, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function sectionSourceQuality(fallbackUsed, fallbackReason) {
  const used = Boolean(fallbackUsed);
  if (used) {
    return {
      className: "supplemented",
      label: "Evidence-Supplemented",
      title: String(
        fallbackReason ||
          "Section points were supplemented from high-confidence evidence because explicit section anchors were sparse."
      ).trim(),
    };
  }
  return {
    className: "anchored",
    label: "Section-Anchored",
    title: "Section points were built from explicit section-anchored evidence.",
  };
}

function formatCompactMethodLine(slot) {
  if (!slot || typeof slot !== "object") return "";
  const label = String(slot.label || slot.slot_key || "").trim();
  const statement = stripConfidenceTag(slot.statement || "");
  if (!label || !statement) return "";
  const refs = asArray(slot.evidence_refs).map((item) => String(item || "").trim()).filter(Boolean);
  return `${label}: ${statement}${refs.length > 0 ? ` [${refs.slice(0, 5).join(", ")}]` : ""}`;
}

function formatCompactSectionLine(slot) {
  if (!slot || typeof slot !== "object") return "";
  const label = String(slot.label || slot.slot_key || "").trim();
  const statement = stripConfidenceTag(slot.statement || "");
  if (!label || !statement) return "";
  const refs = asArray(slot.evidence_refs).map((item) => String(item || "").trim()).filter(Boolean);
  return `${label}: ${statement}${refs.length > 0 ? ` [${refs.slice(0, 5).join(", ")}]` : ""}`;
}

function isEvidenceRefToken(value) {
  const token = String(value || "").trim().toLowerCase();
  if (!token) return false;
  if (token.startsWith("id:")) return true;
  if (token.startsWith("section:")) return true;
  if (token.startsWith("figure:") || token.startsWith("table:")) return true;
  if (/^[ft]\d+[a-z]?$/i.test(token)) return true;
  if (/^supp(?:lement)?[:_\s-]*/i.test(token)) return true;
  return false;
}

function cleanSectionLine(line) {
  return String(line || "")
    .replace(/\[([^\]]+)\]/g, (full, token) => (isEvidenceRefToken(token) ? " " : full))
    .replace(/\s*\((?:source|id|also|refs):[^)]*\)\s*/gi, " ")
    .replace(CONFIDENCE_MARKER_RE, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function extractLineEvidenceRefs(line) {
  const refs = [];
  const seen = new Set();
  const matches = String(line || "").matchAll(/\[([^\]]+)\]/g);
  for (const match of matches) {
    const ref = String(match?.[1] || "").trim();
    if (!isEvidenceRefToken(ref)) continue;
    if (!ref) continue;
    const key = ref.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    refs.push(ref);
  }
  return refs;
}

function lineSourceSummary(line, maxItems = 2) {
  const labels = [];
  const seen = new Set();
  for (const ref of extractLineEvidenceRefs(line)) {
    const label = formatEvidenceRef(ref) || ref;
    const key = String(label || "").toLowerCase().trim();
    if (!key || seen.has(key)) continue;
    seen.add(key);
    labels.push(label);
    if (labels.length >= maxItems) break;
  }
  if (labels.length === 0) return "";
  return labels.join(", ");
}

function sectionEvidenceLabels(lines, maxItems = 24) {
  const labels = [];
  const seen = new Set();
  for (const line of asArray(lines)) {
    for (const ref of extractLineEvidenceRefs(line)) {
      const label = formatEvidenceRef(ref) || ref;
      const key = label.toLowerCase();
      if (!key || seen.has(key)) continue;
      seen.add(key);
      labels.push(label);
      if (labels.length >= maxItems) return labels;
    }
  }
  return labels;
}

function mergeUniqueLabels(primary, secondary, maxItems = 24) {
  const merged = [];
  const seen = new Set();
  for (const value of asArray(primary).concat(asArray(secondary))) {
    const label = String(value || "").trim();
    if (!label) continue;
    const key = label.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    merged.push(label);
    if (merged.length >= maxItems) break;
  }
  return merged;
}

function sectionBlockItems(block) {
  if (!block || typeof block !== "object") return [];
  return asArray(block.items).filter((item) => item && typeof item === "object");
}

function sectionBlockDisplayLines(block, { maxItems = 20, resultsMode = false } = {}) {
  const items = sectionBlockItems(block).slice();
  if (resultsMode) {
    items.sort((left, right) => {
      const leftType = String(left?.result_evidence_type || "");
      const rightType = String(right?.result_evidence_type || "");
      if (leftType === rightType) return 0;
      if (leftType === "text_primary") return -1;
      if (rightType === "text_primary") return 1;
      return 0;
    });
  }
  const lines = items.map((item) => toDisplayLine(item)).filter(Boolean);
  return orderLinesForFlow(keepUniqueByCanonical(lines, Math.max(maxItems * 2, maxItems)), maxItems);
}

function mergeCompactWithDetailLines(
  compactLines,
  detailLines,
  { maxItems = 12, extraLimit = 6 } = {}
) {
  const compact = orderLinesForFlow(keepUniqueByCanonical(asArray(compactLines).filter(Boolean), maxItems), maxItems);
  const compactCanon = new Set(compact.map((line) => canonicalText(linePlainText(line))));
  const extras = [];
  for (const line of orderLinesForFlow(keepUniqueByCanonical(asArray(detailLines).filter(Boolean), maxItems * 2), maxItems * 2)) {
    const plain = linePlainText(line);
    if (!plain || /^n\/a\s*-/i.test(plain)) continue;
    const canon = canonicalText(plain);
    if (!canon || compactCanon.has(canon)) continue;
    compactCanon.add(canon);
    extras.push(line);
    if (extras.length >= extraLimit) break;
  }
  return compact.concat(extras).slice(0, maxItems);
}

function sectionBlockEvidenceLabels(block, maxItems = 24) {
  if (!block || typeof block !== "object") return [];
  const labels = [];
  const seen = new Set();
  const pushLabel = (ref) => {
    const raw = String(ref || "").trim();
    if (!raw) return;
    const label = formatEvidenceRef(raw) || raw;
    const key = label.toLowerCase();
    if (!key || seen.has(key)) return;
    seen.add(key);
    labels.push(label);
  };
  asArray(block.evidence_refs).forEach((ref) => pushLabel(ref));
  for (const item of sectionBlockItems(block)) {
    normalizeEvidenceRefs(item).forEach((ref) => pushLabel(ref));
    if (labels.length >= maxItems) break;
  }
  return labels.slice(0, maxItems);
}

function linePlainText(line) {
  return cleanSectionLine(line)
    .replace(CONFIDENCE_MARKER_RE, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function isMethodLikeLine(line) {
  const text = linePlainText(line).toLowerCase();
  if (!text) return false;
  return /\b(participants?|scanner|acquisition|processing|covariates?|post hoc|seed-based analyses? were conducted|network construction|community detection|was used to test|model(?:ing)?|protocol|procedure|inclusion|exclusion|measured using|analysis was conducted)\b/.test(
    text
  );
}

function isResultLikeLine(line) {
  const text = linePlainText(line).toLowerCase();
  if (!text) return false;
  return /\b(found|showed|demonstrated|revealed|associated|correlated|related to|increased|decreased|higher|lower|difference|effect|significant|dysconnectivity|hyperconnectivity|identified multiple|pattern of connectivity)\b/.test(
    text
  );
}

function hasConcreteOutcomeDetail(line) {
  const text = linePlainText(line).toLowerCase();
  if (!text) return false;
  if (/\bp\s*[<=>]\s*0?\.\d+/.test(text)) return true;
  if (/\b\d+(\.\d+)?\s*(%|ms|mm|sd|ci|or|hr)\b/.test(text)) return true;
  if (/\b(increased|decreased|higher|lower|hyperconnectivity|hypoconnectivity|dysconnectivity|associated with|correlated with|significant)\b/.test(text)) return true;
  return false;
}

function isGenericVisualNarration(line) {
  const text = linePlainText(line).toLowerCase();
  if (!text) return false;
  const looksVisual = /^(figure|table|supplement(?:ary)?\s+figure|supplement(?:ary)?\s+table)\b/.test(text);
  const genericVerb = /\b(shows?|displays?|illustrates?|depicts?|presents?)\b/.test(text);
  if (!looksVisual || !genericVerb) return false;
  return !hasConcreteOutcomeDetail(text);
}

function isIntroLikeLine(line) {
  const text = linePlainText(line).toLowerCase();
  if (!text) return false;
  return /\b(background|objective|aim|rationale|hypothesis|transdiagnostic|reward responsivity|across mood and psychotic disorders)\b/.test(
    text
  );
}

function isConclusionLikeLine(line) {
  const text = linePlainText(line).toLowerCase();
  if (!text) return false;
  return /\b(in conclusion|overall|these findings|this study suggests|supports|implication|conclude|conclusion)\b/.test(
    text
  );
}

function keepUniqueByCanonical(lines, maxItems = 20) {
  const out = [];
  const seen = new Set();
  for (const line of asArray(lines)) {
    const text = String(line || "").trim();
    if (!text) continue;
    const key = canonicalText(linePlainText(text));
    if (!key || seen.has(key)) continue;
    seen.add(key);
    out.push(text);
    if (out.length >= maxItems) break;
  }
  return out;
}

function summarizeAuthorsForDisplay(value) {
  const rawNames = asArray(value)
    .map((item) => String(item || "").trim())
    .filter(Boolean);
  const names = [];
  const seen = new Set();
  for (const name of rawNames) {
    const canonical = name.replace(/\s+/g, " ").trim().toLowerCase();
    if (!canonical) continue;
    if (seen.has(canonical)) {
      if (names.length >= 4) break;
      continue;
    }
    seen.add(canonical);
    names.push(name);
    if (names.length >= 24) break;
  }
  return {
    preview: names.join("; "),
    shownCount: names.length,
    rawCount: rawNames.length,
    hadTruncation: names.length < rawNames.length,
  };
}

function discrepancyReasonText(reasonCode) {
  const key = String(reasonCode || "").toLowerCase().trim();
  if (key === "missing_modality") return "evidence from one or more modalities is missing for this claim";
  if (key === "unsupported") return "the extracted evidence does not sufficiently support this claim";
  if (key === "contradicted" || key === "contradiction") return "evidence packets appear to conflict across modalities";
  if (key === "magnitude_mismatch") return "numerical values appear inconsistent across modalities";
  if (!key) return "cross-modal consistency checks flagged this claim";
  return `reason code: ${key}`;
}

function isEvidenceBackedDiscrepancy(item) {
  if (!item || typeof item !== "object") return false;
  const evidenceCount = Array.isArray(item.evidence) ? item.evidence.length : 0;
  const linkedCount = Array.isArray(item.linked_packet_ids) ? item.linked_packet_ids.length : 0;
  return evidenceCount > 0 || linkedCount > 0;
}

function formatDiscrepancyLine(item) {
  if (!item || typeof item !== "object") return toDisplayLine(item);
  const claim =
    item.claim ||
    item.summary ||
    item.statement ||
    item.finding ||
    item.result ||
    "Potential discrepancy flagged for manual review.";
  const severity = String(item.severity || "medium").toLowerCase();
  const reason = discrepancyReasonText(item.reason);
  const evidenceCount = Array.isArray(item.evidence) ? item.evidence.length : 0;
  const evidenceLabels = asArray(item.evidence).map((value) => formatEvidenceRef(value)).filter(Boolean);
  const linkedCount = Array.isArray(item.linked_packet_ids) ? item.linked_packet_ids.length : 0;
  const evidenceSummary = evidenceLabels.length > 0 ? evidenceLabels.join(", ") : `${evidenceCount} extracted refs`;
  return `Potential discrepancy (manual verification required) [${severity}] ${claim} | Why flagged: ${reason} | Evidence refs: ${evidenceSummary}; linked packets: ${linkedCount}.`;
}

function clarifyUncertaintyLine(line) {
  const text = String(line || "").trim();
  if (!text) return "";
  if (/^unsupported\s*:/i.test(text)) {
    return text.replace(
      /^unsupported\s*:\s*/i,
      "Manual verification required: extracted evidence may not fully support this claim: "
    );
  }
  if (/^missing_modality\s*:/i.test(text)) {
    return text.replace(
      /^missing_modality\s*:\s*/i,
      "Manual verification required: claim lacks cross-modality corroboration: "
    );
  }
  if (/^contradicted\s*:/i.test(text)) {
    return text.replace(
      /^contradicted\s*:\s*/i,
      "Manual verification required: potential cross-modality conflict: "
    );
  }
  if (/^magnitude_mismatch\s*:/i.test(text)) {
    return text.replace(
      /^magnitude_mismatch\s*:\s*/i,
      "Manual verification required: potential numeric mismatch across modalities: "
    );
  }
  return text;
}

function listLines(value) {
  return asArray(value)
    .map((item) => toDisplayLine(item).trim())
    .filter(Boolean);
}

function splitParagraphs(text) {
  return String(text || "")
    .split(/\n{2,}/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function humanizeKey(key) {
  return String(key || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function canonicalText(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/\s+/g, " ")
    .trim();
}

function canonicalStatementKey(value) {
  const cleaned = cleanSectionLine(String(value || ""))
    .toLowerCase()
    .replace(/\.\.\./g, " ");
  const tokens = cleaned.match(CANONICAL_STATEMENT_TOKEN_RE) || [];
  return tokens.join(" ").trim();
}

function areNearDuplicateLines(left, right) {
  const a = canonicalStatementKey(left);
  const b = canonicalStatementKey(right);
  if (!a || !b) return false;
  if (a.includes(b) || b.includes(a)) return true;

  const aTokens = a.split(" ").filter(Boolean);
  const bTokens = b.split(" ").filter(Boolean);
  if (aTokens.length === 0 || bTokens.length === 0) return false;

  const minLen = Math.min(aTokens.length, bTokens.length);
  if (minLen >= 10) {
    let prefixMatches = 0;
    for (let i = 0; i < minLen; i += 1) {
      if (aTokens[i] !== bTokens[i]) break;
      prefixMatches += 1;
    }
    if (prefixMatches / minLen >= 0.86) return true;
  }

  const aSet = new Set(aTokens);
  const bSet = new Set(bTokens);
  let intersection = 0;
  aSet.forEach((token) => {
    if (bSet.has(token)) intersection += 1;
  });
  if (intersection <= 0) return false;
  const overlapMax = intersection / Math.max(aSet.size, bSet.size);
  const overlapMin = intersection / Math.max(1, Math.min(aSet.size, bSet.size));
  return overlapMax >= 0.82 || (overlapMin >= 0.9 && overlapMax >= 0.56);
}

function keepUniqueByNearDuplicate(lines, maxItems = 20) {
  const out = [];
  const seen = [];
  for (const line of asArray(lines)) {
    const text = String(line || "").trim();
    if (!text) continue;
    const key = canonicalStatementKey(text);
    if (!key) continue;
    if (seen.some((existing) => areNearDuplicateLines(existing, key))) continue;
    seen.push(key);
    out.push(text);
    if (out.length >= maxItems) break;
  }
  return out;
}

function categoryTokens(value) {
  return String(value || "")
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function uniqueLines(lines, maxItems = 20) {
  const out = [];
  const seen = new Set();
  for (const line of asArray(lines)) {
    const text = String(line || "").trim();
    if (!text) continue;
    const key = canonicalText(text);
    if (!key || seen.has(key)) continue;
    seen.add(key);
    out.push(text);
    if (out.length >= maxItems) break;
  }
  return out;
}

function normalizeTablePreview(preview) {
  if (!preview || typeof preview !== "object") return null;
  const columns = asArray(preview.columns).map((value) => String(value));
  const rows = asArray(preview.rows)
    .filter((row) => Array.isArray(row))
    .map((row) => row.map((value) => String(value)));
  if (columns.length === 0 && rows.length === 0) return null;
  const totalRows = Number(preview.total_rows);
  const totalCols = Number(preview.total_cols);
  return {
    columns,
    rows,
    totalRows: Number.isFinite(totalRows) ? totalRows : rows.length,
    totalCols: Number.isFinite(totalCols) ? totalCols : columns.length,
  };
}

function compactTablePreview(preview, maxRows = 8, maxCols = 8) {
  const normalized = normalizeTablePreview(preview);
  if (!normalized) return null;
  return {
    ...normalized,
    columns: normalized.columns.slice(0, maxCols),
    rows: normalized.rows.slice(0, maxRows).map((row) => row.slice(0, maxCols)),
  };
}

function firstNumber(value) {
  const match = String(value || "").match(/(\d+)/);
  return match ? Number(match[1]) : Number.POSITIVE_INFINITY;
}

function mediaOrderKey(item, kind = "figure") {
  const candidates = [
    String(item?.[kind === "table" ? "table_id" : "figure_id"] || "").trim(),
    String(item?.anchor || "").trim(),
    String(item?.caption || "").trim(),
  ].filter(Boolean);
  const patterns = kind === "table"
    ? [/\btable\b\s*[_:\s-]*(s?\d+)([a-z]?)/i, /(?:\bfig(?:ure)?\b|^f)\s*[_:\s-]*(s?\d+)([a-z]?)/i]
    : [/(?:\bfig(?:ure)?\b|^f)\s*[_:\s-]*(s?\d+)([a-z]?)/i, /\btable\b\s*[_:\s-]*(s?\d+)([a-z]?)/i];
  for (let idx = 0; idx < candidates.length; idx += 1) {
    const candidate = candidates[idx];
    for (const pattern of patterns) {
      const match = candidate.match(pattern);
      if (!match) continue;
      const raw = String(match[1] || "").toUpperCase();
      const suffix = String(match[2] || "").toLowerCase();
      const suppPenalty = raw.startsWith("S") ? 1 : 0;
      const digitsMatch = raw.match(/(\d+)/);
      const number = digitsMatch ? Number(digitsMatch[1]) : Number.POSITIVE_INFINITY;
      return [suppPenalty, number, suffix || "", idx, candidate.toLowerCase()];
    }
  }
  const fallback = String(item?.anchor || item?.caption || "").toLowerCase();
  const chunk = Number(item?.chunk_id || Number.POSITIVE_INFINITY);
  return [99, chunk, "", 99, fallback];
}

function sortMediaItems(items, kind = "figure") {
  const enriched = asArray(items).map((item, idx) => ({
    item,
    idx,
    key: mediaOrderKey(item, kind),
  }));
  enriched.sort((left, right) => {
    const keyCmp = compareAnchorKeys(left.key, right.key);
    if (keyCmp !== 0) return keyCmp;
    return left.idx - right.idx;
  });
  return enriched.map((entry) => entry.item);
}

function extractAnchorFromLine(line) {
  const text = String(line || "");
  const matches = [...text.matchAll(/\[([^\]]+)\]/g)];
  if (matches.length === 0) return "";
  return String(matches[matches.length - 1][1] || "").trim();
}

function anchorSortKey(anchor) {
  const value = String(anchor || "").toLowerCase().trim();
  if (!value) return [99, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, ""];

  if (value.startsWith("section:")) {
    const parts = value.split(":");
    const sectionIndex = firstNumber(parts[parts.length - 1]);
    const subsectionIndex = firstNumber(parts[parts.length - 2]);
    return [0, subsectionIndex, sectionIndex, value];
  }
  if (value.startsWith("figure:page_")) {
    return [1, firstNumber(value), Number.POSITIVE_INFINITY, value];
  }
  if (value.startsWith("figure:")) {
    return [2, firstNumber(value), Number.POSITIVE_INFINITY, value];
  }
  if (value.startsWith("table:")) {
    return [3, firstNumber(value), Number.POSITIVE_INFINITY, value];
  }
  if (value.startsWith("supp")) {
    return [4, firstNumber(value), Number.POSITIVE_INFINITY, value];
  }
  return [8, firstNumber(value), Number.POSITIVE_INFINITY, value];
}

function compareAnchorKeys(left, right) {
  for (let idx = 0; idx < Math.max(left.length, right.length); idx += 1) {
    const a = left[idx];
    const b = right[idx];
    if (a === b) continue;
    if (a < b) return -1;
    if (a > b) return 1;
  }
  return 0;
}

function orderLinesForFlow(lines, maxItems = 20) {
  const unique = uniqueLines(lines, Math.max(maxItems * 2, maxItems));
  const enriched = unique.map((line, idx) => {
    const anchor = extractAnchorFromLine(line);
    return { line, idx, key: anchorSortKey(anchor) };
  });
  enriched.sort((left, right) => {
    const anchorCmp = compareAnchorKeys(left.key, right.key);
    if (anchorCmp !== 0) return anchorCmp;
    return left.idx - right.idx;
  });
  return enriched.map((item) => item.line).slice(0, maxItems);
}

function methodCategoryRank(line) {
  const match = String(line || "").match(/^\[([^\]]+)\]/);
  const category = match ? String(match[1]).toLowerCase().trim() : "";
  const order = ["methods", "stats", "clinical", "limitations", "ethics", "reproducibility"];
  const idx = order.indexOf(category);
  return idx >= 0 ? idx : order.length + 1;
}

function orderMethodLines(lines, maxItems = 24) {
  const unique = uniqueLines(lines, Math.max(maxItems * 2, maxItems));
  const enriched = unique.map((line, idx) => {
    const anchor = extractAnchorFromLine(line);
    return {
      line,
      idx,
      categoryRank: methodCategoryRank(line),
      anchorKey: anchorSortKey(anchor),
    };
  });
  enriched.sort((left, right) => {
    if (left.categoryRank !== right.categoryRank) {
      return left.categoryRank - right.categoryRank;
    }
    const anchorCmp = compareAnchorKeys(left.anchorKey, right.anchorKey);
    if (anchorCmp !== 0) return anchorCmp;
    return left.idx - right.idx;
  });
  return enriched.map((item) => item.line).slice(0, maxItems);
}

function buildSectionOverview(title, lines) {
  const count = asArray(lines).length;
  if (count === 0) return "";
  const ordered = asArray(lines);
  const firstAnchor = extractAnchorFromLine(ordered[0]);
  const lastAnchor = extractAnchorFromLine(ordered[ordered.length - 1]);
  const anchorPart =
    firstAnchor && lastAnchor
      ? ` from ${firstAnchor} to ${lastAnchor}`
      : firstAnchor
        ? ` starting at ${firstAnchor}`
        : "";
  return `${title} overview: ${count} ordered point${count === 1 ? "" : "s"}${anchorPart}.`;
}

function normalizeSectionBucket(value) {
  const text = String(value || "").toLowerCase().trim();
  if (!text) return "";
  if (/(conclusion|concluding|summary)/.test(text)) return "conclusion";
  if (/(discussion|interpretation|implication|limitations?)/.test(text)) return "discussion";
  if (/(intro|background|objective|aim|rationale|hypoth)/.test(text)) return "introduction";
  if (
    /(results?|finding|outcome|associated|revealed|identified|demonstrated|show(?:ed|s)|effect|dysconnectivity|hyperconnectivity)/.test(
      text
    )
  ) {
    return "results";
  }
  if (/(method|materials|participants|procedure|statistical|analysis|acquisition|design|sample)/.test(text)) return "methods";
  return "";
}

function sectionBucketFromAnchor(anchor) {
  const value = String(anchor || "").trim();
  if (!value) return "";
  const lower = value.toLowerCase();
  if (lower.startsWith("section:")) {
    const parts = value.split(":");
    const sectionName = parts.length > 2 ? parts.slice(1, -1).join(":") : parts[1] || "";
    return normalizeSectionBucket(sectionName);
  }
  if (lower.startsWith("figure:") || lower.startsWith("table:")) return "results";
  if (/^f\d+[a-z]?$/i.test(value) || /^t\d+[a-z]?$/i.test(value)) return "results";
  if (lower.startsWith("supp")) return "results";
  return "";
}

function sectionBucketFromLine(line) {
  const anchorBucket = sectionBucketFromAnchor(extractAnchorFromLine(line));
  if (anchorBucket) return anchorBucket;
  const text = String(line || "").toLowerCase();
  if (!text) return "";
  if (/\b(conclusion|conclude|in summary|overall)\b/.test(text)) return "conclusion";
  if (/\b(discussion|implication|interpretation|contextualize|limitation)\b/.test(text)) return "discussion";
  if (/\b(introduction|background|objective|aim|hypothesis)\b/.test(text)) return "introduction";
  if (
    /\b(result|finding|observed|showed|significant|association|effect|revealed|identified|demonstrated|connectivity pattern)\b/.test(
      text
    )
  ) {
    return "results";
  }
  if (
    /\b(method|materials|participants|procedure|statistical analysis|model|dataset|scanner|covariate|mdmr was used)\b/.test(
      text
    )
  ) {
    return "methods";
  }
  return "";
}

function clampViewerZoom(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 1;
  return Math.max(0.25, Math.min(4, numeric));
}

export default function App() {
  const [workflowStep, setWorkflowStep] = useState("select_source");
  const [nextStepText, setNextStepText] = useState(
    "Step 1: choose a main PDF (or enter URL/DOI), then click Validate."
  );

  const [sourceMode, setSourceMode] = useState("upload");
  const [url, setUrl] = useState("");
  const [doi, setDoi] = useState("");
  const [mainFile, setMainFile] = useState(null);
  const [supplementFiles, setSupplementFiles] = useState([]);

  const [validation, setValidation] = useState({
    valid: null,
    message: "Select source input, then click Validate Input.",
  });

  const [statusPayload, setStatusPayload] = useState(null);
  const [backendReady, setBackendReady] = useState(false);
  const [connectionState, setConnectionState] = useState("starting");

  const [jobs, setJobs] = useState([]);
  const [selectedJobId, setSelectedJobId] = useState(null);
  const selectedJob = useMemo(
    () => jobs.find((item) => item.job_id === selectedJobId) || null,
    [jobs, selectedJobId]
  );
  const latestCompletedJob = useMemo(
    () => jobs.find((item) => item.status === "completed") || null,
    [jobs]
  );

  const [reportSummary, setReportSummary] = useState(null);
  const [reportPayload, setReportPayload] = useState(null);
  const [documentMedia, setDocumentMedia] = useState({ figures: [], tables: [] });
  const [mediaError, setMediaError] = useState("");
  const [detailViewOpen, setDetailViewOpen] = useState(false);
  const [assetViewer, setAssetViewer] = useState(null);

  const [submitBusy, setSubmitBusy] = useState(false);
  const [loadBusy, setLoadBusy] = useState(false);
  const [saveBusy, setSaveBusy] = useState(false);
  const [reportSaved, setReportSaved] = useState(false);
  const [runtimeInfo, setRuntimeInfo] = useState(null);
  const [lastError, setLastError] = useState("");

  const [eventBar, setEventBar] = useState({ level: "info", text: "Starting desktop bootstrap...", ts: nowLabel() });
  const [eventHistory, setEventHistory] = useState([]);

  const lastRuntimeSinceRef = useRef(null);
  const eventDedupeRef = useRef(new Map());
  const selectedJobIdRef = useRef(null);
  const jobStatusCacheRef = useRef(new Map());
  const mainFileInputRef = useRef(null);
  const supplementsInputRef = useRef(null);

  const processing = statusPayload?.processing || {};
  const inflight = Number(processing.inflight || 0);
  const workerCapacity = Number(processing.worker_capacity || 1);

  const canSubmit = backendReady && validation.valid && !submitBusy;
  const canLoadReport = !!latestCompletedJob && !loadBusy && backendReady;
  const isPaused = Boolean(processing.paused);
  const validationClass =
    validation.valid === true ? "ok" : validation.valid === false ? "bad" : "neutral";
  const structuredSummary =
    reportPayload?.summary_json && typeof reportPayload.summary_json === "object"
      ? reportPayload.summary_json
      : null;
  const executiveParagraphs = splitParagraphs(
    stripConfidenceTag(structuredSummary?.executive_summary || reportPayload?.summary || "")
  );
  const keyFindingLines = orderLinesForFlow(listLines(structuredSummary?.key_findings), 20);
  const methodsCardLines = orderLinesForFlow(
    keepUniqueByNearDuplicate(listLines(reportSummary?.methods_card).map((line) => cleanSectionLine(line)), 16),
    8
  );
  const sectionsCardLines = orderLinesForFlow(
    keepUniqueByNearDuplicate(listLines(reportSummary?.sections_card).map((line) => cleanSectionLine(line)), 24),
    12
  );
  const rerunRecommended = Boolean(reportSummary?.rerun_recommended);
  const reportCapabilities = reportSummary?.report_capabilities && typeof reportSummary.report_capabilities === "object"
    ? reportSummary.report_capabilities
    : {};
  const reportModelUsage =
    reportPayload?.analysis_diagnostics?.diagnostics?.model_usage &&
    typeof reportPayload.analysis_diagnostics.diagnostics.model_usage === "object"
      ? reportPayload.analysis_diagnostics.diagnostics.model_usage
      : null;
  const modalityEntries =
    structuredSummary?.modalities && typeof structuredSummary.modalities === "object"
      ? Object.entries(structuredSummary.modalities)
      : [];
  const rawDiscrepancies = asArray(structuredSummary?.discrepancies)
    .filter((item) => item && typeof item === "object")
    .filter((item) => isEvidenceBackedDiscrepancy(item));
  const discrepancyLines = rawDiscrepancies.length > 0
    ? orderLinesForFlow(
      rawDiscrepancies
        .slice()
        .sort((left, right) => {
          const order = { high: 0, medium: 1, low: 2 };
          const leftRank = order[String(left?.severity || "medium").toLowerCase()] ?? 1;
          const rightRank = order[String(right?.severity || "medium").toLowerCase()] ?? 1;
          if (leftRank !== rightRank) return leftRank - rightRank;
          return 0;
        })
        .map((item) => formatDiscrepancyLine(item)),
      24
    )
    : orderLinesForFlow(
      listLines(asArray(structuredSummary?.discrepancies).filter((item) => typeof item !== "object")),
      24
    );
  const methodsStrengthLines = orderLinesForFlow(listLines(structuredSummary?.methods_strengths), 24);
  const methodsWeaknessLines = orderLinesForFlow(listLines(structuredSummary?.methods_weaknesses), 24);
  const reproducibilityLines = orderLinesForFlow(
    listLines(structuredSummary?.reproducibility_ethics || structuredSummary?.reproducibility_and_ethics),
    16
  );
  const uncertaintyLines = orderLinesForFlow(
    listLines(structuredSummary?.uncertainty_gaps || structuredSummary?.uncertainty_and_gaps).map((line) =>
      clarifyUncertaintyLine(line)
    ),
    20
  );
  const paperMeta =
    structuredSummary?.paper_meta && typeof structuredSummary.paper_meta === "object"
      ? structuredSummary.paper_meta
      : structuredSummary?.paper_metadata && typeof structuredSummary.paper_metadata === "object"
        ? structuredSummary.paper_metadata
        : null;
  const paperMetaEntries = paperMeta
    ? Object.entries(paperMeta).filter(([, value]) => value !== null && value !== undefined && String(value) !== "")
    : [];
  const authorSummary = summarizeAuthorsForDisplay(paperMeta?.authors);
  const apiRoot = useMemo(() => API_BASE.replace(/\/api\/?$/, ""), []);
  const toApiMediaUrl = (path) => {
    if (!path) return "";
    if (String(path).startsWith("http://") || String(path).startsWith("https://")) return String(path);
    return `${apiRoot}${String(path).startsWith("/") ? "" : "/"}${path}`;
  };
  const mediaFigures = asArray(documentMedia?.figures);
  const mediaTables = asArray(documentMedia?.tables);
  const tableLikeFigures = dedupeMediaItems(
    mediaFigures.filter((item) => !isSupplementMediaItem(item) && isTableLikeMediaItem(item))
  );
  const mainFigures = sortMediaItems(
    dedupeMediaItems(mediaFigures.filter((item) => !isSupplementMediaItem(item) && !isTableLikeMediaItem(item))),
    "figure"
  );
  const suppFigures = sortMediaItems(
    dedupeMediaItems(
      mediaFigures
        .filter((item) => isSupplementMediaItem(item))
        .concat(mediaTables.filter((item) => isSupplementMediaItem(item)))
    ),
    "figure"
  );
  const allTables = sortMediaItems(
    dedupeMediaItems(
      mediaTables
        .filter((item) => !isSupplementMediaItem(item))
        .concat(tableLikeFigures)
    ),
    "table"
  );
  const totalMediaAssets = mainFigures.length + suppFigures.length + allTables.length;
  const mediaOverview = totalMediaAssets > 0
    ? `Assets overview: ${totalMediaAssets} visual item${totalMediaAssets === 1 ? "" : "s"} grouped in reading order.`
    : "";
  const suppFigureExpectedRefs = asArray(structuredSummary?.coverage?.supp_figures?.expected_refs)
    .map((value) => formatEvidenceRef(value) || String(value || "").trim())
    .filter(Boolean);
  const suppTableExpectedRefs = asArray(structuredSummary?.coverage?.supp_tables?.expected_refs)
    .map((value) => formatEvidenceRef(value) || String(value || "").trim())
    .filter(Boolean);
  const suppExpectedRefs = keepUniqueByCanonical(
    suppFigureExpectedRefs.concat(suppTableExpectedRefs),
    48
  );
  const assetSections = [
    { title: "Tables", items: allTables },
    { title: "Figures", items: mainFigures },
    { title: "Supplementary Figures", items: suppFigures },
  ];

  const textModalityPackets = asArray(structuredSummary?.modalities?.text?.findings).filter(
    (item) => item && typeof item === "object"
  );
  const inferredMethodDetails = textModalityPackets
    .filter((item) => {
      const tokens = categoryTokens(item.category || "other");
      const preferred = ["methods", "stats", "reproducibility", "ethics", "limitations", "clinical"];
      return tokens.some((token) => preferred.includes(token));
    })
    .map((item) => {
      const statement = item.statement || item.summary || item.claim || item.result || "";
      const evidence = asArray(item.evidence_refs).slice(0, 5).join(", ");
      const tokens = categoryTokens(item.category || "other");
      const preferred = ["methods", "stats", "reproducibility", "ethics", "limitations", "clinical"];
      const normalizedCategory = tokens.find((token) => preferred.includes(token)) || String(item.category || "");
      const category = normalizedCategory ? `[${normalizedCategory}] ` : "";
      return `${category}${statement}${evidence ? ` [${evidence}]` : ""}`;
    });
  const methodologyDetailLines = orderMethodLines(
    asArray(structuredSummary?.methodology_details).map((item) => toDisplayLine(item)).concat(inferredMethodDetails),
    24
  );
  const discrepancyOverview = discrepancyLines.length > 0
    ? `Discrepancy review: ${discrepancyLines.length} potential item${discrepancyLines.length === 1 ? "" : "s"}. These are model-flagged checks and must be manually verified against the source paper before drawing conclusions.`
    : "";
  const methodsOverview = buildSectionOverview("Methods", methodologyDetailLines);
  const methodsStrengthOverview = buildSectionOverview("Method strengths", methodsStrengthLines);
  const methodsWeaknessOverview = buildSectionOverview("Method weaknesses", methodsWeaknessLines);
  const reproducibilityOverview = buildSectionOverview("Reproducibility and ethics", reproducibilityLines);
  const uncertaintyOverview = buildSectionOverview("Uncertainty and gaps", uncertaintyLines);
  const displayMethodologyLines = methodologyDetailLines.map((line) => stripMethodsLabel(line));
  const displayMethodStrengthLines = methodsStrengthLines.map((line) => stripMethodsLabel(line));
  const displayMethodWeaknessLines = methodsWeaknessLines.map((line) => stripMethodsLabel(line));
  const displayReproducibilityLines = reproducibilityLines.map((line) => stripMethodsLabel(line));
  const methodsCompactRows = asArray(structuredSummary?.methods_compact)
    .filter((slot) => slot && typeof slot === "object")
    .slice(0, 12)
    .map((slot) => ({
      line: formatCompactMethodLine(slot),
      status: String(slot.status || "not_found").toLowerCase(),
    }))
    .filter((row) => row.line);
  const methodsCompactRowsDisplay = [];
  const compactSeen = [];
  for (const row of methodsCompactRows) {
    if (!row?.line) continue;
    const key = canonicalStatementKey(row.line);
    if (!key) continue;
    if (compactSeen.some((existing) => areNearDuplicateLines(existing, key))) continue;
    compactSeen.push(key);
    methodsCompactRowsDisplay.push(row);
  }
  const showMethodsCompactNarrative = methodsCompactRowsDisplay.length > 0 && !Boolean(reportCapabilities.presentation_evidence);
  const sectionsCompactMap =
    structuredSummary?.sections_compact && typeof structuredSummary.sections_compact === "object"
      ? structuredSummary.sections_compact
      : null;
  const compactRowsForSection = (sectionKey, maxItems) =>
    asArray(sectionsCompactMap?.[sectionKey])
      .filter((slot) => slot && typeof slot === "object")
      .slice(0, maxItems)
      .map((slot) => ({
        line: formatCompactSectionLine(slot),
        status: String(slot.status || "not_found").toLowerCase(),
      }))
      .filter((row) => row.line);
  const introductionCompactRows = compactRowsForSection("introduction", 3);
  const resultsCompactRows = compactRowsForSection("results", 5);
  const discussionCompactRows = compactRowsForSection("discussion", 4);
  const conclusionCompactRows = compactRowsForSection("conclusion", 2);
  const hasCompactSections =
    introductionCompactRows.length > 0 ||
    resultsCompactRows.length > 0 ||
    discussionCompactRows.length > 0 ||
    conclusionCompactRows.length > 0;
  const figureModalityPackets = asArray(structuredSummary?.modalities?.figure?.findings).filter(
    (item) => item && typeof item === "object"
  );
  const tableModalityPackets = asArray(structuredSummary?.modalities?.table?.findings).filter(
    (item) => item && typeof item === "object"
  );
  const supplementModalityPackets = asArray(structuredSummary?.modalities?.supplement?.findings).filter(
    (item) => item && typeof item === "object"
  );
  const sectionBuckets = {
    introduction: [],
    methods: [],
    results: [],
    conclusion: [],
    discussion: [],
  };
  if (!hasCompactSections) {
    for (const packet of textModalityPackets) {
      const line = toDisplayLine(packet);
      if (!line) continue;
      const fromAnchor = sectionBucketFromAnchor(packet?.anchor);
      const fromCategory = normalizeSectionBucket(packet?.category);
      const fromText = sectionBucketFromLine(line);
      const bucket = fromAnchor || fromText || fromCategory || "";
      if (bucket && sectionBuckets[bucket]) {
        sectionBuckets[bucket].push(line);
      }
    }
  }
  const textDisplayLines = textModalityPackets.map((item) => toDisplayLine(item)).filter(Boolean);

  const introFallbackLines = textDisplayLines.filter((line) => isIntroLikeLine(line) && !isMethodLikeLine(line));
  const introSeedLines = textDisplayLines
    .slice(0, 8)
    .filter((line) => !isMethodLikeLine(line) && !isResultLikeLine(line));
  const introductionSectionLines = hasCompactSections
    ? introductionCompactRows.map((row) => row.line)
    : orderLinesForFlow(
      keepUniqueByCanonical(sectionBuckets.introduction.filter((line) => !isMethodLikeLine(line)).concat(introFallbackLines, introSeedLines), 32),
      14
    );

  const methodsCompactCandidateLines = Boolean(reportCapabilities.presentation_evidence)
    ? []
    : methodsCompactRowsDisplay.map((row) => row.line);
  const methodsCandidates = sectionBuckets.methods
    .concat(methodsCompactCandidateLines)
    .concat(displayMethodologyLines)
    .concat(displayMethodStrengthLines)
    .concat(displayMethodWeaknessLines)
    .filter((line) => {
      const bucket = sectionBucketFromLine(line);
      if (bucket === "results" && !isMethodLikeLine(line)) return false;
      return isMethodLikeLine(line) || bucket === "methods";
    });
  const methodsSectionLines = orderMethodLines(keepUniqueByNearDuplicate(methodsCandidates, 40), 24);

  const mediaResultLines = figureModalityPackets
    .map((item) => toDisplayLine(item))
    .concat(tableModalityPackets.map((item) => toDisplayLine(item)))
    .concat(supplementModalityPackets.map((item) => toDisplayLine(item)));

  const textResultLines = keepUniqueByCanonical(
    sectionBuckets.results
      .concat(textDisplayLines.filter((line) => isResultLikeLine(line)))
      .filter((line) => {
        const bucket = sectionBucketFromLine(line);
        if (bucket === "introduction" || bucket === "discussion" || bucket === "conclusion") return false;
        if (isMethodLikeLine(line) && !hasConcreteOutcomeDetail(line)) return false;
        return true;
      }),
    40
  );
  const mediaResultLinesQualified = keepUniqueByCanonical(
    mediaResultLines.filter((line) => !isGenericVisualNarration(line) || hasConcreteOutcomeDetail(line)),
    30
  );

  const resultsSectionLines = hasCompactSections
    ? resultsCompactRows.map((row) => row.line)
    : orderLinesForFlow(
      keepUniqueByCanonical(
        textResultLines
          .concat(mediaResultLinesQualified)
          .filter((line) => {
            const bucket = sectionBucketFromLine(line);
            if (bucket === "introduction" || bucket === "discussion" || bucket === "conclusion") return false;
            if (isMethodLikeLine(line) && !hasConcreteOutcomeDetail(line)) return false;
            return true;
          }),
        50
      ),
      30
    );

  const inferredConclusionLines = sectionBuckets.discussion
    .filter((line) => isConclusionLikeLine(line) && !isMethodLikeLine(line));
  const conclusionCandidates = sectionBuckets.conclusion
    .filter((line) => !isMethodLikeLine(line))
    .concat(inferredConclusionLines);
  const conclusionSectionLines = hasCompactSections
    ? conclusionCompactRows.map((row) => row.line)
    : orderLinesForFlow(keepUniqueByCanonical(conclusionCandidates, 20), 12);

  const discussionCandidates = sectionBuckets.discussion.filter((line) => {
      const bucket = sectionBucketFromLine(line);
      if (bucket === "methods" && !isConclusionLikeLine(line)) return false;
      return true;
    });
  const discussionSectionLines = hasCompactSections
    ? discussionCompactRows.map((row) => row.line)
    : orderLinesForFlow(
      keepUniqueByCanonical(discussionCandidates, 28),
      18
    );
  const introductionEmptyMessage =
    hasCompactSections
      ? "No introduction slots extracted."
      : sectionBuckets.introduction.length === 0
      ? "No explicit Introduction heading was detected in extracted text; showing best available background/objective lines."
      : "No introduction evidence extracted.";
  const conclusionEmptyMessage =
    hasCompactSections
      ? "No conclusion slots extracted."
      : sectionBuckets.conclusion.length === 0
      ? "No explicit Conclusion heading was detected in extracted text."
      : "No conclusion evidence extracted.";
  const introductionEvidenceLabels = sectionEvidenceLabels(introductionSectionLines);
  const methodsEvidenceLabels = sectionEvidenceLabels(methodsSectionLines);
  const resultsEvidenceLabels = sectionEvidenceLabels(resultsSectionLines);
  const conclusionEvidenceLabels = sectionEvidenceLabels(conclusionSectionLines);
  const discussionEvidenceLabels = sectionEvidenceLabels(discussionSectionLines);
  const backendSections =
    structuredSummary?.sections && typeof structuredSummary.sections === "object"
      ? structuredSummary.sections
      : null;
  const hasBackendSections = Boolean(backendSections);
  const useCompactSections = hasCompactSections && !hasBackendSections;
  const backendIntroduction = hasBackendSections ? backendSections?.introduction : null;
  const backendMethods = hasBackendSections ? backendSections?.methods : null;
  const backendResults = hasBackendSections ? backendSections?.results : null;
  const backendConclusion = hasBackendSections ? backendSections?.conclusion : null;
  const backendDiscussion = hasBackendSections ? backendSections?.discussion : null;
  const introductionDetailLines = hasBackendSections ? sectionBlockDisplayLines(backendIntroduction, { maxItems: 14 }) : [];
  const methodsDetailLines = hasBackendSections ? sectionBlockDisplayLines(backendMethods, { maxItems: 24 }) : [];
  const resultsDetailLines = hasBackendSections
    ? sectionBlockDisplayLines(backendResults, { maxItems: 30, resultsMode: true })
    : [];
  const conclusionDetailLines = hasBackendSections ? sectionBlockDisplayLines(backendConclusion, { maxItems: 12 }) : [];
  const discussionDetailLines = hasBackendSections ? sectionBlockDisplayLines(backendDiscussion, { maxItems: 18 }) : [];

  const introductionDisplayLines = useCompactSections
    ? mergeCompactWithDetailLines(introductionSectionLines, introductionDetailLines, { maxItems: 10, extraLimit: 7 })
    : hasBackendSections
      ? introductionDetailLines
    : introductionSectionLines;
  const methodsDisplayLinesRaw = hasBackendSections
    ? methodsDetailLines
    : methodsSectionLines;
  const methodsDisplayLines = orderMethodLines(keepUniqueByNearDuplicate(methodsDisplayLinesRaw, 24), 24);
  const resultsDisplayLines = useCompactSections
    ? mergeCompactWithDetailLines(resultsSectionLines, resultsDetailLines, { maxItems: 24, extraLimit: 20 })
    : hasBackendSections
      ? resultsDetailLines
    : resultsSectionLines;
  const conclusionDisplayLines = useCompactSections
    ? mergeCompactWithDetailLines(conclusionSectionLines, conclusionDetailLines, { maxItems: 8, extraLimit: 6 })
    : hasBackendSections
      ? conclusionDetailLines
    : conclusionSectionLines;
  const discussionDisplayLines = useCompactSections
    ? mergeCompactWithDetailLines(discussionSectionLines, discussionDetailLines, { maxItems: 10, extraLimit: 6 })
    : hasBackendSections
      ? discussionDetailLines
    : discussionSectionLines;

  const introductionDisplayEvidenceLabels = useCompactSections
    ? mergeUniqueLabels(sectionBlockEvidenceLabels(backendIntroduction), sectionEvidenceLabels(introductionDisplayLines))
    : hasBackendSections
      ? sectionBlockEvidenceLabels(backendIntroduction)
    : introductionEvidenceLabels;
  const methodsDisplayEvidenceLabels = hasBackendSections
    ? mergeUniqueLabels(sectionBlockEvidenceLabels(backendMethods), sectionEvidenceLabels(methodsDisplayLines))
    : methodsEvidenceLabels;
  const resultsDisplayEvidenceLabels = useCompactSections
    ? mergeUniqueLabels(sectionBlockEvidenceLabels(backendResults), sectionEvidenceLabels(resultsDisplayLines))
    : hasBackendSections
      ? sectionBlockEvidenceLabels(backendResults)
    : resultsEvidenceLabels;
  const conclusionDisplayEvidenceLabels = useCompactSections
    ? mergeUniqueLabels(sectionBlockEvidenceLabels(backendConclusion), sectionEvidenceLabels(conclusionDisplayLines))
    : hasBackendSections
      ? sectionBlockEvidenceLabels(backendConclusion)
    : conclusionEvidenceLabels;
  const discussionDisplayEvidenceLabels = useCompactSections
    ? mergeUniqueLabels(sectionBlockEvidenceLabels(backendDiscussion), sectionEvidenceLabels(discussionDisplayLines))
    : hasBackendSections
      ? sectionBlockEvidenceLabels(backendDiscussion)
    : discussionEvidenceLabels;

  const introductionFallbackUsed = hasBackendSections ? Boolean(backendIntroduction?.fallback_used) : false;
  const methodsFallbackUsed = hasBackendSections ? Boolean(backendMethods?.fallback_used) : false;
  const resultsFallbackUsed = hasBackendSections ? Boolean(backendResults?.fallback_used) : false;
  const conclusionFallbackUsed = hasBackendSections ? Boolean(backendConclusion?.fallback_used) : false;
  const discussionFallbackUsed = hasBackendSections ? Boolean(backendDiscussion?.fallback_used) : false;

  const introductionFallbackReason = hasBackendSections ? String(backendIntroduction?.fallback_reason || "").trim() : "";
  const methodsFallbackReason = hasBackendSections ? String(backendMethods?.fallback_reason || "").trim() : "";
  const resultsFallbackReason = hasBackendSections ? String(backendResults?.fallback_reason || "").trim() : "";
  const conclusionFallbackReason = hasBackendSections ? String(backendConclusion?.fallback_reason || "").trim() : "";
  const discussionFallbackReason = hasBackendSections ? String(backendDiscussion?.fallback_reason || "").trim() : "";
  const introductionSourceQuality = sectionSourceQuality(introductionFallbackUsed, introductionFallbackReason);
  const methodsSourceQuality = sectionSourceQuality(methodsFallbackUsed, methodsFallbackReason);
  const resultsSourceQuality = sectionSourceQuality(resultsFallbackUsed, resultsFallbackReason);
  const conclusionSourceQuality = sectionSourceQuality(conclusionFallbackUsed, conclusionFallbackReason);
  const discussionSourceQuality = sectionSourceQuality(discussionFallbackUsed, discussionFallbackReason);

  const introductionDisplayEmptyMessage = hasBackendSections
    ? introductionFallbackReason || "No introduction evidence extracted."
    : introductionEmptyMessage;
  const conclusionDisplayEmptyMessage = hasBackendSections
    ? conclusionFallbackReason || "No conclusion evidence extracted."
    : conclusionEmptyMessage;

  useEffect(() => {
    selectedJobIdRef.current = selectedJobId;
  }, [selectedJobId]);

  const markSourceDirty = () => {
    setValidation({ valid: null, message: "Input updated. Click Validate Input to continue." });
    setNextStepText("Step 2: click Validate Input.");
    if (workflowStep !== "select_source") {
      setWorkflowStep("select_source");
    }
  };

  const pushEvent = (text, level = "info", dedupeKey = null, timestamp = null) => {
    if (dedupeKey) {
      const prev = eventDedupeRef.current.get(dedupeKey);
      if (prev && Date.now() - prev < 2_000) {
        return;
      }
      eventDedupeRef.current.set(dedupeKey, Date.now());
    }
    const payload = { level, text, ts: timestamp ? formatEtDateTime(timestamp) : nowLabel() };
    setEventBar(payload);
    setEventHistory((prev) => [payload, ...prev].slice(0, 20));
  };

  const validateInput = () => {
    if (sourceMode === "url") {
      const { resolvedUrl, normalizedDoi } = resolveUrlAndDoiInputs(url, doi);
      if (!resolvedUrl && !normalizedDoi) {
        setValidation({
          valid: false,
          message: "URL or DOI is required. Add one in Step 1 and validate again.",
        });
        setWorkflowStep("select_source");
        setNextStepText("Add URL or DOI, then click Validate.");
        return false;
      }

      if (resolvedUrl && !/^https?:\/\//i.test(resolvedUrl) && !normalizedDoi) {
        setValidation({
          valid: false,
          message: "Enter a full URL (https://...) or DOI (10.xxxx/...).",
        });
        setWorkflowStep("select_source");
        setNextStepText("Correct URL/DOI format, then click Validate.");
        return false;
      }

      setUrl(resolvedUrl);
      setDoi(normalizedDoi);
      setValidation({ valid: true, message: "Validation passed for URL/DOI source." });
      setWorkflowStep("validate");
      if (backendReady) {
        setNextStepText("Step 3: click Submit Analysis.");
      } else {
        setNextStepText("Validation passed. Waiting for backend reconnect to enable submit.");
      }
      return true;
    }

    if (!mainFile) {
      setValidation({
        valid: false,
        message: "Main PDF is required. Choose a file in Step 1.",
      });
      setWorkflowStep("select_source");
      setNextStepText("Choose the main PDF and validate again.");
      return false;
    }

    const invalidSupp = supplementFiles.find((file) => {
      const lower = file.name.toLowerCase();
      return !SUPPLEMENT_EXTS.some((ext) => lower.endsWith(ext));
    });
    if (invalidSupp) {
      setValidation({
        valid: false,
        message: `Unsupported supplement type: ${invalidSupp.name}.`,
      });
      setWorkflowStep("select_source");
      setNextStepText("Remove unsupported supplement files, then validate again.");
      return false;
    }

    setValidation({ valid: true, message: "Validation passed for upload source." });
    setWorkflowStep("validate");
    if (backendReady) {
      setNextStepText("Step 3: click Submit Analysis.");
    } else {
      setNextStepText("Validation passed. Waiting for backend reconnect to enable submit.");
    }
    return true;
  };

  const refreshRuntimeInfo = async () => {
    try {
      const info = await getDesktopRuntimeInfo();
      setRuntimeInfo(info);
    } catch {
      // Ignore when running outside tauri dev shell.
    }
  };

  const refreshStatusAndJobs = async () => {
    const [statusRes, jobsRes] = await Promise.all([
      fetchStatus(),
      fetchJobs({ sort: "updated_at:desc", limit: 250, offset: 0 }),
    ]);

    setStatusPayload(statusRes);
    setBackendReady(true);
    setConnectionState("ready");
    setLastError("");

    const nextJobs = Array.isArray(jobsRes.items) ? jobsRes.items : [];
    const nextJobIds = new Set(nextJobs.map((item) => item.job_id));
    const cache = jobStatusCacheRef.current;
    for (const [jobId] of cache.entries()) {
      if (!nextJobIds.has(jobId)) {
        cache.delete(jobId);
      }
    }
    for (const item of nextJobs) {
      const prev = cache.get(item.job_id);
      const failedNow =
        item.status === "failed" &&
        (!prev || prev.status !== "failed" || prev.message !== item.message);
      if (failedNow) {
        const reason = item.message || "No error details were provided.";
        pushEvent(`Job #${item.job_id} failed: ${reason}`, "error", `job-failed-${item.job_id}-${item.updated_at || ""}`);
        setLastError(reason);
      }
      cache.set(item.job_id, {
        status: item.status,
        message: item.message || "",
        updated_at: item.updated_at || "",
      });
    }
    setJobs(nextJobs);
    const selected = selectedJobIdRef.current;
    if (!selected && nextJobs.length > 0) {
      setSelectedJobId(nextJobs[0].job_id);
    }
    if (selected && !nextJobs.find((item) => item.job_id === selected)) {
      setSelectedJobId(nextJobs[0]?.job_id || null);
    }

    const running = nextJobs.some((item) => item.status === "running");
    return running ? 1000 : 4000;
  };

  const refreshRuntimeEvents = async () => {
    try {
      const payload = await fetchRuntimeEvents({ since: lastRuntimeSinceRef.current, limit: 25 });
      const items = Array.isArray(payload.events) ? payload.events : [];
      if (items.length === 0) {
        return;
      }
      lastRuntimeSinceRef.current = items[items.length - 1].timestamp;
      items.forEach((item) => {
        pushEvent(
          formatEvent(item),
          item.severity || "info",
          item.event_id || `${item.kind}:${item.timestamp}`,
          item.timestamp || null
        );
      });
    } catch {
      // Runtime event feed should not break UI polling.
    }
  };

  useEffect(() => {
    let cancelled = false;
    let timer = null;

    const bootstrap = async () => {
      setConnectionState("starting");
      try {
        const payload = await fetchBootstrap();
        if (cancelled) return;
        setStatusPayload({
          backend_ready: payload.backend_ready,
          processing: payload.processing,
          model_exists: payload.models?.model_exists,
          mmproj_exists: payload.models?.mmproj_exists,
        });
        setBackendReady(Boolean(payload.backend_ready));
        setConnectionState(payload.backend_ready ? "ready" : "unavailable");
        setJobs(Array.isArray(payload.latest_jobs) ? payload.latest_jobs : []);
        if ((payload.latest_jobs || []).length > 0) {
          setSelectedJobId(payload.latest_jobs[0].job_id);
        }
        pushEvent("Desktop bootstrap loaded.", "info", "bootstrap-loaded");
      } catch (err) {
        if (cancelled) return;
        setConnectionState("unavailable");
        setBackendReady(false);
        if (err instanceof DesktopApiError) {
          const msg = toUiErrorMessage(err);
          setLastError(msg);
          pushEvent(msg, "error", "bootstrap-error");
        } else {
          const msg = String(err?.message || err || "Bootstrap failed");
          setLastError(msg);
          pushEvent(msg, "error", "bootstrap-error-raw");
        }
      } finally {
        await refreshRuntimeInfo();
      }
    };

    const pollLoop = async () => {
      if (cancelled) return;
      let nextDelay = 3000;
      try {
        nextDelay = await refreshStatusAndJobs();
        await refreshRuntimeEvents();
      } catch (err) {
        setBackendReady(false);
        setConnectionState("unavailable");
        const msg = toUiErrorMessage(err);
        setLastError(msg);
        pushEvent(`Backend unavailable. ${msg}`, "error", "poll-backend-unavailable");
        nextDelay = 3000;
      }
      if (!cancelled) {
        timer = setTimeout(pollLoop, nextDelay);
      }
    };

    bootstrap().finally(() => {
      if (!cancelled) {
        timer = setTimeout(pollLoop, 1200);
      }
    });

    return () => {
      cancelled = true;
      if (timer) {
        clearTimeout(timer);
      }
    };
  }, []);

  useEffect(() => {
    if (!selectedJob) {
      return;
    }

    if (selectedJob.status === "completed") {
      setWorkflowStep((prev) => (prev === "review" ? prev : "monitoring"));
      setNextStepText("Selected job is completed. Click Load Latest Report.");
      return;
    }
    if (selectedJob.status === "failed") {
      setWorkflowStep("monitoring");
      setNextStepText("Selected job failed. Fix source input and submit a new run.");
      return;
    }
    if (selectedJob.status === "running") {
      setWorkflowStep("monitoring");
      setNextStepText("Job is running. Wait for completion, then load report.");
      return;
    }
    if (selectedJob.status === "queued") {
      setWorkflowStep("submitted");
      setNextStepText("Job is queued. Monitor Active Work.");
    }
  }, [selectedJob]);

  const handleSubmit = async () => {
    const valid = validateInput();
    if (!valid) {
      pushEvent("Validation failed. Resolve issues and retry.", "warning", "validate-failed");
      return;
    }
    if (!backendReady) {
      pushEvent("Backend unavailable. Retry connection before submission.", "error", "submit-no-backend");
      return;
    }

    setSubmitBusy(true);
    try {
      let result;
      if (sourceMode === "url") {
        const { resolvedUrl, normalizedDoi } = resolveUrlAndDoiInputs(url, doi);
        result = await submitFromUrl(resolvedUrl, normalizedDoi || null);
      } else {
        result = await submitFromUpload(mainFile, supplementFiles);
      }
      setWorkflowStep("submitted");
      setNextStepText("Job submitted. Monitor progress in Active Work.");
      pushEvent(`Submitted document ${result.document_id} (job ${result.job_id}).`, "info", "submit-success");

      const jobsPayload = await fetchJobs({ sort: "updated_at:desc", limit: 250, offset: 0 });
      const nextJobs = Array.isArray(jobsPayload.items) ? jobsPayload.items : [];
      setJobs(nextJobs);
      setSelectedJobId(result.job_id);
    } catch (err) {
      const msg = toUiErrorMessage(err);
      pushEvent(`Submit failed: ${msg}`, "error", "submit-failed");
      setLastError(msg);
      setValidation({
        valid: false,
        message: `Submission failed: ${msg}`,
      });
      setWorkflowStep("select_source");
      setNextStepText("Submission failed. Adjust source input and validate again.");
    } finally {
      setSubmitBusy(false);
    }
  };

  const handleClearSourceInputs = () => {
    setUrl("");
    setDoi("");
    setMainFile(null);
    setSupplementFiles([]);
    setSourceMode("upload");
    setValidation({
      valid: null,
      message: "Source inputs cleared. Add URL/DOI or files, then validate.",
    });
    setWorkflowStep("select_source");
    setNextStepText("Step 1: choose a main PDF (or enter URL/DOI), then click Validate.");
    if (mainFileInputRef.current) {
      mainFileInputRef.current.value = "";
    }
    if (supplementsInputRef.current) {
      supplementsInputRef.current.value = "";
    }
    pushEvent("Source inputs cleared.", "info", "source-inputs-cleared");
  };

  const handleLoadReport = async () => {
    const targetJob = selectedJob?.status === "completed" ? selectedJob : latestCompletedJob;
    if (!targetJob) {
      pushEvent("Load Report requires at least one completed job.", "warning", "load-not-completed");
      return;
    }

    setLoadBusy(true);
    try {
      if (selectedJobId !== targetJob.job_id) {
        setSelectedJobId(targetJob.job_id);
      }
      const [summary, full, mediaResult, saveStatus] = await Promise.all([
        fetchReportSummary(targetJob.document_id),
        fetchReport(targetJob.document_id),
        fetchDocumentMedia(targetJob.document_id)
          .then((payload) => ({ ok: true, payload }))
          .catch((error) => ({ ok: false, error })),
        fetchReportSaveStatus(targetJob.document_id).catch(() => ({ saved: false })),
      ]);
      const media = mediaResult?.ok ? mediaResult.payload : { figures: [], tables: [] };
      setReportSummary(summary);
      setReportPayload(full);
      setReportSaved(Boolean(saveStatus?.saved));
      setDocumentMedia({
        figures: Array.isArray(media?.figures) ? media.figures : [],
        tables: Array.isArray(media?.tables) ? media.tables : [],
      });
      if (mediaResult?.ok) {
        const figureCount = Array.isArray(media?.figures) ? media.figures.length : 0;
        const tableCount = Array.isArray(media?.tables) ? media.tables.length : 0;
        pushEvent(
          `Loaded media assets: ${figureCount} figure item${figureCount === 1 ? "" : "s"}, ${tableCount} table item${tableCount === 1 ? "" : "s"}.`,
          "info",
          "load-media-counts"
        );
      }
      if (!mediaResult?.ok) {
        const reason = toUiErrorMessage(mediaResult?.error || "Unknown media error.");
        setMediaError(`Visual media links are unavailable for this report. ${reason}`);
      } else if (Array.isArray(media?.figures) || Array.isArray(media?.tables)) {
        setMediaError("");
      } else {
        setMediaError("Visual media links are unavailable for this report.");
      }
      setWorkflowStep("review");
      setNextStepText("Report loaded. Review modality sections and discrepancies.");
      pushEvent(
        `Loaded latest completed report for job ${targetJob.job_id} (document ${targetJob.document_id}).`,
        "info",
        `load-report-success-${targetJob.job_id}`
      );
    } catch (err) {
      const msg = toUiErrorMessage(err);
      setLastError(msg);
      setDocumentMedia({ figures: [], tables: [] });
      setReportSaved(false);
      setMediaError("");
      pushEvent(`Failed to load report: ${msg}`, "error", "load-report-failed");
    } finally {
      setLoadBusy(false);
    }
  };

  const handleSaveReport = async () => {
    if (!selectedJob || selectedJob.status !== "completed") {
      pushEvent("Save Report requires a completed job selection.", "warning", "save-not-completed");
      return;
    }
    setSaveBusy(true);
    try {
      const payload = await saveReport(selectedJob.document_id);
      setReportSaved(true);
      pushEvent(
        `Saved report for document ${selectedJob.document_id} at ${payload?.saved_path || "saved_reports folder"}.`,
        "info",
        "save-report-success"
      );
    } catch (err) {
      const msg = toUiErrorMessage(err);
      pushEvent(`Save report failed: ${msg}`, "error", "save-report-failed");
      setLastError(msg);
    } finally {
      setSaveBusy(false);
    }
  };

  const handleRetryBackend = async () => {
    try {
      setConnectionState("starting");
      pushEvent("Retrying backend startup...", "info", "retry-backend");
      await restartBackend();
      await refreshStatusAndJobs();
      pushEvent("Backend retry command completed.", "info", "retry-backend-complete");
      await refreshRuntimeInfo();
    } catch (err) {
      const msg = toUiErrorMessage(err);
      setConnectionState("unavailable");
      pushEvent(`Backend retry failed: ${msg}`, "error", "retry-backend-failed");
      setLastError(msg);
    }
  };

  const handleOpenLogs = async () => {
    try {
      await openLogsFolder();
    } catch {
      pushEvent("Open Logs is only available inside the desktop shell runtime.", "warning", "open-logs-unavailable");
    }
  };

  const handleExportReport = () => {
    if (!reportSummary?.export_url) {
      pushEvent("Export URL is not available.", "warning", "export-missing");
      return;
    }
    const link = reportSummary.export_url.startsWith("http")
      ? reportSummary.export_url
      : `${API_BASE.replace(/\/api$/, "")}${reportSummary.export_url}`;
    window.open(link, "_blank", "noopener,noreferrer");
  };

  const handleOpenDetailView = () => {
    if (!reportPayload) {
      pushEvent("Load a report first to open detailed analysis.", "warning", "detail-view-no-report");
      return;
    }
    setAssetViewer(null);
    setDetailViewOpen(true);
  };

  const closeDetailView = () => {
    setAssetViewer(null);
    setDetailViewOpen(false);
  };

  const openAssetViewer = (src, title, kindHint = "") => {
    if (!src) return;
    const kind = String(kindHint || "").trim() || mediaViewerKind(src);
    setAssetViewer({ kind, src, title: String(title || "Asset"), zoom: kind === "image" ? 1 : undefined });
  };

  const adjustImageZoom = (delta) => {
    setAssetViewer((prev) => {
      if (!prev || prev.kind !== "image") return prev;
      return { ...prev, zoom: clampViewerZoom((prev.zoom || 1) + delta) };
    });
  };

  const resetImageZoom = () => {
    setAssetViewer((prev) => {
      if (!prev || prev.kind !== "image") return prev;
      return { ...prev, zoom: 1 };
    });
  };

  const openTableViewer = (title, tablePreview, sourceHref = "") => {
    const normalized = normalizeTablePreview(tablePreview);
    if (!normalized) return;
    setAssetViewer({
      kind: "table",
      title: String(title || "Table"),
      tablePreview: normalized,
      sourceHref: String(sourceHref || ""),
    });
  };

  const handleProcessingAction = async (action) => {
    try {
      if (action === "pause") {
        await pauseProcessing();
        pushEvent("Processing paused.", "info", "processing-pause");
      } else if (action === "resume") {
        await resumeProcessing();
        pushEvent("Processing resumed.", "info", "processing-resume");
      } else if (action === "cleanup") {
        const payload = await cleanOrphans();
        pushEvent(`Cleaned ${payload.killed || 0} orphan workers.`, "info", "processing-cleanup");
      } else if (action === "recover") {
        const payload = await recoverProcessing();
        pushEvent(
          `Recovered ${payload.recovered_jobs || 0} stale jobs.`,
          "info",
          "processing-recover"
        );
      }
      await refreshStatusAndJobs();
    } catch (err) {
      const msg = toUiErrorMessage(err);
      pushEvent(`Processing action failed: ${msg}`, "error", `processing-${action}-fail`);
      setLastError(msg);
    }
  };

  const handleDiagnosticsExport = async () => {
    try {
      const path = await exportDiagnosticsBundle();
      pushEvent(`Diagnostics bundle created at ${path}.`, "info", "diag-export");
    } catch (err) {
      const msg = String(err?.message || err || "Diagnostics export failed.");
      pushEvent(msg, "error", "diag-export-failed");
    }
  };

  return (
    <div className="app-shell">
      <header className="top-bar">
        <div>
          <h1>PaperEval Desktop V8</h1>
          <p>Guided multimodal paper evaluation with deterministic lifecycle control. Build marker: HF3.</p>
        </div>
        <div className="top-actions">
          <button onClick={handleOpenLogs}>Open Logs</button>
          <button onClick={handleRetryBackend}>Retry Backend</button>
        </div>
      </header>

      <section className="health-bar card">
        <div className="health-item">
          <label>Backend</label>
          <strong className={backendReady ? "ok" : "bad"}>
            {connectionState === "starting" ? "Starting" : backendReady ? "Ready" : "Unavailable"}
          </strong>
        </div>
        <div className="health-item">
          <label>Models</label>
          <strong className={statusPayload?.model_exists && statusPayload?.mmproj_exists ? "ok" : "bad"}>
            {statusPayload?.model_exists && statusPayload?.mmproj_exists ? "Ready" : "Missing"}
          </strong>
        </div>
        <div className="health-item">
          <label>Processing</label>
          <strong>{isPaused ? "Paused" : processing.running ? "Running" : "Stopped"}</strong>
        </div>
        <div className="health-item">
          <label>Inflight</label>
          <strong>
            {inflight}/{workerCapacity}
          </strong>
        </div>
      </section>

      <section className="connection-card card">
        <div className="section-head">Connection</div>
        <div className="connection-body">
          <div className={`connection-badge ${backendReady ? "ok" : connectionState === "starting" ? "warn" : "bad"}`}>
            {backendReady ? "Backend Ready" : connectionState === "starting" ? "Backend Starting" : "Backend Unavailable"}
          </div>
          <div className="connection-text">
            {backendReady
              ? "Workflow actions are enabled."
              : "Source selection and validation remain available while reconnecting."}
          </div>
        </div>
      </section>

      <main className="workspace-grid">
        <section className="card panel workflow-panel">
          <div className="section-head">Workflow</div>
          <div className="workflow-scroll">
            <div className="next-step">Next step: {nextStepText}</div>

            <div className="step-list">
              {WORKFLOW_STEPS.map((step, idx) => {
                const activeIdx = WORKFLOW_STEPS.findIndex((item) => item.id === workflowStep);
                const state = idx < activeIdx ? "done" : idx === activeIdx ? "active" : "pending";
                return (
                  <div key={step.id} className={`step-item ${state}`}>
                    <span>{idx + 1}. {step.label}</span>
                    <small>{state}</small>
                  </div>
                );
              })}
            </div>

            <div className="subsection">
              <h3>Step 1: Select Source</h3>
              <label className="radio-row">
                <input
                  type="radio"
                  checked={sourceMode === "url"}
                  onChange={() => {
                    setSourceMode("url");
                    setValidation({ valid: null, message: "Source changed. Click Validate Input to continue." });
                    setWorkflowStep("select_source");
                    setNextStepText("Enter URL/DOI and click Validate.");
                  }}
                />
                <span>From URL / DOI</span>
              </label>
              <label className="radio-row">
                <input
                  type="radio"
                  checked={sourceMode === "upload"}
                  onChange={() => {
                    setSourceMode("upload");
                    setValidation({ valid: null, message: "Source changed. Click Validate Input to continue." });
                    setWorkflowStep("select_source");
                    setNextStepText("Choose a main PDF and click Validate.");
                  }}
                />
                <span>From Upload</span>
              </label>

              <div className="field-grid">
                <label>
                  URL
                  <input
                    type="text"
                    placeholder="https://doi.org/..."
                    value={url}
                    onChange={(event) => {
                      setSourceMode("url");
                      setUrl(event.target.value);
                      markSourceDirty();
                    }}
                  />
                </label>
                <label>
                  DOI
                  <input
                    type="text"
                    placeholder="10.1000/xyz123"
                    value={doi}
                    onChange={(event) => {
                      setSourceMode("url");
                      setDoi(event.target.value);
                      markSourceDirty();
                    }}
                  />
                </label>
                <label>
                  Main PDF
                  <input
                    ref={mainFileInputRef}
                    type="file"
                    accept="application/pdf"
                    onChange={(event) => {
                      setSourceMode("upload");
                      setMainFile(event.target.files?.[0] || null);
                      markSourceDirty();
                    }}
                  />
                </label>
                <label>
                  Supplements
                  <input
                    ref={supplementsInputRef}
                    type="file"
                    multiple
                    onChange={(event) => {
                      setSourceMode("upload");
                      setSupplementFiles(Array.from(event.target.files || []));
                      markSourceDirty();
                    }}
                  />
                </label>
              </div>
              <div className="source-action-row">
                <button type="button" onClick={handleClearSourceInputs}>
                  Clear Inputs
                </button>
              </div>
              <div className="hint">Tip: entering URL/DOI switches to URL mode; selecting files switches to Upload mode.</div>
              {mainFile && <div className="hint">Main PDF: {mainFile.name}</div>}
              {supplementFiles.length > 0 && <div className="hint">Supplements: {supplementFiles.length} selected</div>}
            </div>

            <div className="subsection">
              <h3>Step 2: Validate Input</h3>
              <div className={`validation ${validationClass}`}>{validation.message}</div>
              <button onClick={validateInput}>Validate</button>
            </div>

            <div className="subsection">
              <h3>Step 3: Submit</h3>
              <button disabled={!canSubmit} onClick={handleSubmit}>
                {submitBusy ? "Submitting..." : "Submit Analysis"}
              </button>
              {!backendReady && <div className="hint">Submission is disabled while backend is unavailable.</div>}
            </div>

            <div className="subsection controls-grid">
              <h3>Processing Controls</h3>
              <button disabled={!backendReady || isPaused} onClick={() => handleProcessingAction("pause")}>Pause Processing</button>
              <button disabled={!backendReady || !isPaused} onClick={() => handleProcessingAction("resume")}>Resume Processing</button>
              <button disabled={!backendReady} onClick={() => handleProcessingAction("cleanup")}>Clean Worker Orphans</button>
              <button disabled={!backendReady} onClick={() => handleProcessingAction("recover")}>Recover Processing State</button>
            </div>
          </div>
        </section>

        <section className="card panel active-panel">
          <div className="section-head">Active Work</div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Job</th>
                  <th>Source</th>
                  <th>Status</th>
                  <th>Progress</th>
                  <th>Updated (ET)</th>
                </tr>
              </thead>
              <tbody>
                {jobs.length === 0 && (
                  <tr>
                    <td colSpan={5} className="empty-row">No jobs yet. Submit analysis to populate this table.</td>
                  </tr>
                )}
                {jobs.map((item) => (
                  <tr
                    key={item.job_id}
                    className={selectedJobId === item.job_id ? "selected" : ""}
                    onClick={() => setSelectedJobId(item.job_id)}
                  >
                    <td>#{item.job_id} / doc {item.document_id}</td>
                    <td>{sourceKindLabel(item.source_kind)}</td>
                    <td>{item.status}</td>
                    <td>{Math.round((item.progress || 0) * 100)}%</td>
                    <td>{item.updated_at ? formatEtDateTime(item.updated_at) : "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="subsection">
            <h3>Selected Job Details</h3>
            {!selectedJob && <p className="empty">No job selected.</p>}
            {selectedJob && (
              <>
                <div className="detail-grid">
                  <div>Job ID</div><div>{selectedJob.job_id}</div>
                  <div>Document ID</div><div>{selectedJob.document_id}</div>
                  <div>Status</div><div>{selectedJob.status}</div>
                  <div>Progress</div><div>{Math.round((selectedJob.progress || 0) * 100)}%</div>
                  <div>Message</div><div>{selectedJob.message || "-"}</div>
                </div>
                {selectedJob.status === "failed" && (
                  <div className="failure-callout">
                    <strong>Failure Reason</strong>
                    <p>{selectedJob.message || "No error details were provided by backend."}</p>
                    <p className="hint">
                      Next step: if this DOI/URL is publisher-blocked, switch to From Upload and choose the main PDF.
                    </p>
                  </div>
                )}
              </>
            )}
          </div>
        </section>

        <section className="card panel report-panel">
          <div className="section-head">Report</div>
          <div className="report-scroll">
            <div className="button-row">
              <button disabled={!canLoadReport} onClick={handleLoadReport}>
                {loadBusy ? "Loading..." : "Load Latest Report"}
              </button>
              <button disabled={!reportPayload || saveBusy || reportSaved} onClick={handleSaveReport}>
                {saveBusy ? "Saving..." : reportSaved ? "Report Saved" : "Save Report"}
              </button>
              <button disabled={!reportSummary?.export_url} onClick={handleExportReport}>Export Report JSON</button>
              <button disabled={!reportPayload} onClick={handleOpenDetailView}>Open Detailed Analysis</button>
            </div>

            <div className="report-status">
              Report: {reportSummary?.report_status === "ready" ? "Ready" : selectedJob?.status === "failed" ? "Failed" : "Not Ready"}
            </div>
            {rerunRecommended && (
              <div className="failure-inline">
                Legacy report detected. Re-run analysis to enable deterministic section-fidelity compact outputs.
              </div>
            )}
            {mediaError && <div className="failure-inline">{mediaError}</div>}
            {selectedJob?.status === "failed" && (
              <div className="failure-inline">
                {selectedJob.message || "Analysis failed before report generation."}
              </div>
            )}

            <div className="subsection">
              <h3>Key Findings Snapshot</h3>
              {keyFindingLines.length === 0 && <p className="empty">No key findings available.</p>}
              {keyFindingLines.length > 0 && (
                <ol className="detail-list detail-list-ordered">
                  {keyFindingLines.slice(0, 6).map((line, idx) => (
                    <li key={`kfs-${idx}`}>{line}</li>
                  ))}
                </ol>
              )}
            </div>

            <div className="subsection">
              <h3>Summary Cards</h3>
              {!reportSummary && <p className="empty">No report loaded yet.</p>}
              {methodsCardLines.length > 0 && (
                <div className="modality-card" style={{ marginBottom: 8 }}>
                  <h4>METHODS AT A GLANCE</h4>
                  {methodsCardLines.map((line, idx) => (
                    <p key={`methods-card-${idx}`}>{line}</p>
                  ))}
                </div>
              )}
              {sectionsCardLines.length > 0 && (
                <div className="modality-card" style={{ marginBottom: 8 }}>
                  <h4>SECTION SNAPSHOT</h4>
                  {sectionsCardLines.map((line, idx) => (
                    <p key={`sections-card-${idx}`}>{line}</p>
                  ))}
                </div>
              )}
              {reportSummary?.modality_cards?.length > 0 && (
                <div className="summary-cards-scroll">
                  <div className="modality-grid">
                    {reportSummary.modality_cards.map((card) => (
                      <div key={card.modality} className="modality-card">
                        <h4>{String(card.modality || "modality").toUpperCase()}</h4>
                        <div>{card.finding_count || 0} findings</div>
                        {(card.highlights || []).slice(0, 3).map((item, idx) => (
                          <p key={`${card.modality}-h-${idx}`}>{item}</p>
                        ))}
                        {(card.coverage_gaps || []).slice(0, 2).map((item, idx) => (
                          <p key={`${card.modality}-g-${idx}`} className="coverage-gap">{item}</p>
                        ))}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {Object.keys(reportCapabilities).length > 0 && (
              <div className="subsection">
                <h3>Report Capabilities</h3>
                <div className="detail-grid">
                  <div>Methods compact</div><div>{String(Boolean(reportCapabilities.methods_compact))}</div>
                  <div>Sections compact</div><div>{String(Boolean(reportCapabilities.sections_compact))}</div>
                  <div>Coverage snapshot line</div><div>{String(Boolean(reportCapabilities.coverage_snapshot_line))}</div>
                </div>
              </div>
            )}

            <div className="subsection">
              <h3>Discrepancies</h3>
              <div>{reportSummary ? reportSummary.discrepancy_count : 0} flagged discrepancies</div>
            </div>

            <div className="subsection">
              <h3>Diagnostics</h3>
              <div className="button-row">
                <button onClick={refreshRuntimeInfo}>Refresh Runtime</button>
                <button onClick={handleDiagnosticsExport}>Export Diagnostics</button>
              </div>
              <div className="diag-block">
                <div>API Base: {API_BASE}</div>
                <div>Backend Ready: {String(backendReady)}</div>
                <div>Desktop PID: {runtimeInfo?.desktop_pid || runtimeInfo?.desktop_instance_pid || "unknown"}</div>
                <div>Backend PID: {runtimeInfo?.backend_pid || "unknown"}</div>
                <div>UI Python: {runtimeInfo?.ui_python_path || "unknown"}</div>
                <div>Backend Python: {runtimeInfo?.backend_python_path || "unknown"}</div>
                {reportModelUsage && (
                  <>
                    <div>Text model calls: {reportModelUsage.text_calls ?? 0}</div>
                    <div>Reasoning model calls: {reportModelUsage.deep_calls ?? 0}</div>
                    <div>Vision model calls: {reportModelUsage.vision_calls ?? 0}</div>
                  </>
                )}
                {lastError && <div className="error-text">Last error: {lastError}</div>}
              </div>
            </div>

            <div className="subsection">
              <h3>Recent Events (ET)</h3>
              <ul className="events-list">
                {eventHistory.slice(0, 6).map((event, idx) => (
                  <li key={`${event.ts}-${idx}`} className={event.level}>
                    <span className="event-time">{event.ts}</span>
                    <span>{event.text}</span>
                  </li>
                ))}
              </ul>
            </div>

            {reportPayload?.summary_json && (
              <div className="subsection">
                <h3>Summary JSON Preview</h3>
                <pre>{JSON.stringify(reportPayload.summary_json, null, 2).slice(0, 2000)}</pre>
              </div>
            )}
          </div>
        </section>
      </main>

      {detailViewOpen && (
        <div className="detail-modal-backdrop" onClick={closeDetailView}>
          <section className="detail-modal card" onClick={(event) => event.stopPropagation()}>
            <header className="detail-modal-header">
              <h3>Detailed Analysis</h3>
              <button onClick={closeDetailView}>Close</button>
            </header>
            <div className="detail-modal-body">
              {!reportPayload && <p className="empty">No report loaded yet.</p>}
              {reportPayload && (
                <>
                  {rerunRecommended && (
                    <div className="failure-inline">
                      Legacy report detected. Re-run analysis for deterministic section-fidelity output.
                    </div>
                  )}
                  <div className="subsection">
                    <h3>Executive Summary</h3>
                    {executiveParagraphs.length === 0 && <p className="empty">No executive summary available.</p>}
                    {executiveParagraphs.map((paragraph, idx) => (
                      <p key={`exec-${idx}`}>{paragraph}</p>
                    ))}
                  </div>

                  {paperMetaEntries.length > 0 && (
                    <div className="subsection">
                      <h3>Paper Metadata</h3>
                      <div className="detail-meta-grid">
                        {paperMetaEntries.map(([key, value]) => (
                          <div key={key} className="detail-meta-row">
                            <strong>{humanizeKey(key)}</strong>
                            {String(key).toLowerCase() === "authors" ? (
                              <span>
                                {authorSummary.preview || toDisplayLine(value)}
                                {authorSummary.hadTruncation && (
                                  <details className="detail-evidence-footnote">
                                    <summary>
                                      Show extracted author list ({authorSummary.rawCount})
                                    </summary>
                                    <div>{asArray(value).map((item) => String(item || "").trim()).filter(Boolean).join("; ")}</div>
                                  </details>
                                )}
                              </span>
                            ) : (
                              <span>{toDisplayLine(value)}</span>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="subsection">
                    <h3>Introduction</h3>
                    <div
                      className={`section-source-quality-badge ${introductionSourceQuality.className}`}
                      title={introductionSourceQuality.title}
                    >
                      {introductionSourceQuality.label}
                    </div>
                    {introductionDisplayLines.length === 0 && <p className="empty">{introductionDisplayEmptyMessage}</p>}
                    {introductionDisplayLines.length > 0 && (
                      <ol className="detail-list detail-list-ordered">
                        {introductionDisplayLines.map((line, idx) => {
                          const sourceSummary = lineSourceSummary(line);
                          return (
                            <li key={`intro-${idx}`}>
                              {cleanSectionLine(line)}
                              {sourceSummary ? <span className="detail-inline-source"> (source: {sourceSummary})</span> : null}
                            </li>
                          );
                        })}
                      </ol>
                    )}
                    {introductionDisplayEvidenceLabels.length > 0 && (
                      <details className="detail-evidence-footnote">
                        <summary>Evidence refs ({introductionDisplayEvidenceLabels.length})</summary>
                        <div>{introductionDisplayEvidenceLabels.join(", ")}</div>
                      </details>
                    )}
                  </div>

                  <div className="subsection">
                    <h3>Methods</h3>
                    <div
                      className={`section-source-quality-badge ${methodsSourceQuality.className}`}
                      title={methodsSourceQuality.title}
                    >
                      {methodsSourceQuality.label}
                    </div>
                    {showMethodsCompactNarrative && (
                      <>
                        <strong>Methods At a Glance</strong>
                        <ol className="detail-list detail-list-ordered">
                          {methodsCompactRowsDisplay.map((row, idx) => (
                            <li
                              key={`methods-compact-${idx}`}
                              className={row.status === "found" ? "" : "detail-list-item-muted"}
                            >
                              {row.line}
                            </li>
                          ))}
                        </ol>
                        <strong>Additional Method Notes</strong>
                      </>
                    )}
                    {methodsDisplayLines.length === 0 && <p className="empty">{methodsFallbackReason || "No methods evidence extracted."}</p>}
                    {methodsDisplayLines.length > 0 && (
                      <ol className="detail-list detail-list-ordered">
                        {methodsDisplayLines.map((line, idx) => {
                          const sourceSummary = lineSourceSummary(line);
                          return (
                            <li key={`methods-${idx}`}>
                              {cleanSectionLine(line)}
                              {sourceSummary ? <span className="detail-inline-source"> (source: {sourceSummary})</span> : null}
                            </li>
                          );
                        })}
                      </ol>
                    )}
                    {methodsDisplayEvidenceLabels.length > 0 && (
                      <details className="detail-evidence-footnote">
                        <summary>Evidence refs ({methodsDisplayEvidenceLabels.length})</summary>
                        <div>{methodsDisplayEvidenceLabels.join(", ")}</div>
                      </details>
                    )}
                  </div>

                  <div className="subsection">
                    <h3>Results</h3>
                    <div
                      className={`section-source-quality-badge ${resultsSourceQuality.className}`}
                      title={resultsSourceQuality.title}
                    >
                      {resultsSourceQuality.label}
                    </div>
                    {resultsDisplayLines.length === 0 && totalMediaAssets === 0 && <p className="empty">{resultsFallbackReason || "No results evidence extracted."}</p>}
                    {resultsDisplayLines.length > 0 && (
                      <ol className="detail-list detail-list-ordered">
                        {resultsDisplayLines.map((line, idx) => {
                          const sourceSummary = lineSourceSummary(line);
                          return (
                            <li key={`results-${idx}`}>
                              {cleanSectionLine(line)}
                              {sourceSummary ? <span className="detail-inline-source"> (source: {sourceSummary})</span> : null}
                            </li>
                          );
                        })}
                      </ol>
                    )}
                    {resultsDisplayEvidenceLabels.length > 0 && (
                      <details className="detail-evidence-footnote">
                        <summary>Evidence refs ({resultsDisplayEvidenceLabels.length})</summary>
                        <div>{resultsDisplayEvidenceLabels.join(", ")}</div>
                      </details>
                    )}
                    {(mainFigures.length > 0 || suppFigures.length > 0 || allTables.length > 0) && (
                      <>
                        <strong>Embedded Figures and Tables</strong>
                        {mediaOverview && <p className="detail-overview">{mediaOverview}</p>}
                        <div className="detail-asset-lanes">
                          {assetSections.map(({ title, items }) => (
                            <div key={title} className="detail-modality-card detail-asset-lane">
                              <h4>{title}</h4>
                              {items.length === 0 && title === "Supplementary Figures" && suppExpectedRefs.length > 0 && (
                                <p className="empty">
                                  {`No supplementary files were loaded. Supplementary refs mentioned in text (${suppExpectedRefs.length}): ${suppExpectedRefs.join(", ")}.`}
                                </p>
                              )}
                              {items.length === 0 && !(title === "Supplementary Figures" && suppExpectedRefs.length > 0) && <p className="empty">No assets.</p>}
                              {items.length > 0 && (
                                <div className="detail-asset-row-scroll">
                                  <ul className="detail-asset-row">
                                    {items.map((item, idx) => {
                                      const proxiedImageHref = item.image_url ? toApiMediaUrl(item.image_url) : "";
                                      const proxiedSourceHref = item.source_proxy_url
                                        ? toApiMediaUrl(item.source_proxy_url)
                                        : item.asset_url
                                          ? toApiMediaUrl(item.asset_url)
                                          : "";
                                      const assetHref = item.asset_url ? toApiMediaUrl(item.asset_url) : proxiedSourceHref;
                                      const sourceHref = proxiedSourceHref || assetHref || proxiedImageHref;
                                      const imageHref = proxiedImageHref || (isImageHref(sourceHref) ? sourceHref : "");
                                      const hasOpenableAsset = Boolean(sourceHref || imageHref);
                                      const tablePreview = normalizeTablePreview(item.table_preview);
                                      const compactPreview = compactTablePreview(item.table_preview, 8, 8);
                                      const label = assetLabel(item, `asset ${idx + 1}`);
                                      return (
                                        <li key={`${title}-${item.chunk_id || idx}`} className="detail-asset-card">
                                          <div className="detail-asset-title" title={item.caption || item.anchor || `asset ${idx + 1}`}>
                                            {label}
                                          </div>
                                          <div className="detail-asset-meta">
                                            <span>{item.anchor || "anchor:n/a"}</span>
                                            {item.page ? <span>Page {item.page}</span> : null}
                                            {imageHref ? (
                                              <button
                                                type="button"
                                                className="detail-link-button"
                                                onClick={() => openAssetViewer(imageHref, item.caption || item.anchor || title, "image")}
                                              >
                                                Open image
                                              </button>
                                            ) : sourceHref ? (
                                              <button
                                                type="button"
                                                className="detail-link-button"
                                                onClick={() => openAssetViewer(sourceHref, item.caption || item.anchor || title)}
                                              >
                                                Open asset
                                              </button>
                                            ) : (
                                              <span>No preview file</span>
                                            )}
                                            {sourceHref ? (
                                              <a className="detail-link-anchor" href={sourceHref} target="_blank" rel="noreferrer">
                                                Open source
                                              </a>
                                            ) : null}
                                            {tablePreview && (
                                              <button
                                                type="button"
                                                className="detail-link-button"
                                                onClick={() =>
                                                  openTableViewer(item.caption || item.anchor || title, tablePreview, sourceHref)
                                                }
                                              >
                                                Open table view
                                              </button>
                                            )}
                                          </div>
                                          {imageHref && (
                                            <button
                                              type="button"
                                              className="detail-icon-button"
                                              onClick={() => openAssetViewer(imageHref, item.caption || item.anchor || title, "image")}
                                            >
                                              <img
                                                className="detail-asset-icon"
                                                src={imageHref}
                                                alt={item.caption || item.anchor || title}
                                                onError={(event) => {
                                                  if (proxiedImageHref && event.currentTarget.src !== proxiedImageHref) {
                                                    event.currentTarget.src = proxiedImageHref;
                                                  }
                                                }}
                                              />
                                              <span>Enlarge</span>
                                            </button>
                                          )}
                                          {!imageHref && hasOpenableAsset && (
                                            <button
                                              type="button"
                                              className="detail-icon-button"
                                              onClick={() => openAssetViewer(sourceHref, item.caption || item.anchor || title)}
                                            >
                                              <div className="detail-asset-icon detail-asset-icon-placeholder">
                                                {isPdfHref(sourceHref) ? "PDF" : "ASSET"}
                                              </div>
                                              <span>{isPdfHref(sourceHref) ? "Open PDF" : "Open asset"}</span>
                                            </button>
                                          )}
                                          {compactPreview && (
                                            <details className="detail-inline-preview">
                                              <summary>Quick table preview</summary>
                                              <div className="detail-table-preview">
                                                <div className="detail-table-caption">
                                                  Preview ({compactPreview.totalRows} rows, {compactPreview.totalCols} columns)
                                                </div>
                                                <table>
                                                  {compactPreview.columns.length > 0 && (
                                                    <thead>
                                                      <tr>
                                                        {compactPreview.columns.map((col, colIdx) => (
                                                          <th key={`${item.chunk_id || idx}-col-${colIdx}`}>{col}</th>
                                                        ))}
                                                      </tr>
                                                    </thead>
                                                  )}
                                                  <tbody>
                                                    {compactPreview.rows.map((row, rowIdx) => (
                                                      <tr key={`${item.chunk_id || idx}-row-${rowIdx}`}>
                                                        {row.map((cell, cellIdx) => (
                                                          <td key={`${item.chunk_id || idx}-cell-${rowIdx}-${cellIdx}`}>{cell}</td>
                                                        ))}
                                                      </tr>
                                                    ))}
                                                  </tbody>
                                                </table>
                                              </div>
                                            </details>
                                          )}
                                        </li>
                                      );
                                    })}
                                  </ul>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </>
                    )}
                  </div>

                  <div className="subsection">
                    <h3>Conclusion</h3>
                    <div
                      className={`section-source-quality-badge ${conclusionSourceQuality.className}`}
                      title={conclusionSourceQuality.title}
                    >
                      {conclusionSourceQuality.label}
                    </div>
                    {conclusionDisplayLines.length === 0 && <p className="empty">{conclusionDisplayEmptyMessage}</p>}
                    {conclusionDisplayLines.length > 0 && (
                      <ol className="detail-list detail-list-ordered">
                        {conclusionDisplayLines.map((line, idx) => {
                          const sourceSummary = lineSourceSummary(line);
                          return (
                            <li key={`conclusion-${idx}`}>
                              {cleanSectionLine(line)}
                              {sourceSummary ? <span className="detail-inline-source"> (source: {sourceSummary})</span> : null}
                            </li>
                          );
                        })}
                      </ol>
                    )}
                    {conclusionDisplayEvidenceLabels.length > 0 && (
                      <details className="detail-evidence-footnote">
                        <summary>Evidence refs ({conclusionDisplayEvidenceLabels.length})</summary>
                        <div>{conclusionDisplayEvidenceLabels.join(", ")}</div>
                      </details>
                    )}
                  </div>

                  <div className="subsection">
                    <h3>Discussion</h3>
                    <div
                      className={`section-source-quality-badge ${discussionSourceQuality.className}`}
                      title={discussionSourceQuality.title}
                    >
                      {discussionSourceQuality.label}
                    </div>
                    {discussionDisplayLines.length === 0 && <p className="empty">{discussionFallbackReason || "No discussion evidence extracted."}</p>}
                    {discussionDisplayLines.length > 0 && (
                      <ol className="detail-list detail-list-ordered">
                        {discussionDisplayLines.map((line, idx) => {
                          const sourceSummary = lineSourceSummary(line);
                          return (
                            <li key={`discussion-${idx}`}>
                              {cleanSectionLine(line)}
                              {sourceSummary ? <span className="detail-inline-source"> (source: {sourceSummary})</span> : null}
                            </li>
                          );
                        })}
                      </ol>
                    )}
                    {discussionDisplayEvidenceLabels.length > 0 && (
                      <details className="detail-evidence-footnote">
                        <summary>Evidence refs ({discussionDisplayEvidenceLabels.length})</summary>
                        <div>{discussionDisplayEvidenceLabels.join(", ")}</div>
                      </details>
                    )}
                  </div>

                  {(discrepancyLines.length > 0 || uncertaintyLines.length > 0) && (
                    <details className="subsection">
                      <summary>Quality Checks</summary>
                      {discrepancyOverview && <p className="detail-overview">{discrepancyOverview}</p>}
                      {discrepancyLines.length > 0 && (
                        <>
                          <strong>Discrepancy Flags (Manual Verification Required)</strong>
                          <ol className="detail-list detail-list-ordered">
                            {discrepancyLines.map((line, idx) => (
                              <li key={`disc-${idx}`}>{line}</li>
                            ))}
                          </ol>
                        </>
                      )}
                      {uncertaintyLines.length > 0 && (
                        <>
                          <strong>Uncertainty and Coverage Gaps</strong>
                          <ol className="detail-list detail-list-ordered">
                            {uncertaintyLines.map((line, idx) => (
                              <li key={`unc-${idx}`}>{line}</li>
                            ))}
                          </ol>
                        </>
                      )}
                    </details>
                  )}

                  {structuredSummary && (
                    <details className="subsection">
                      <summary>Raw Structured JSON</summary>
                      <pre>{JSON.stringify(structuredSummary, null, 2)}</pre>
                    </details>
                  )}
                </>
              )}
            </div>
          </section>
        </div>
      )}

      {assetViewer && (
        <div className="detail-image-backdrop" onClick={() => setAssetViewer(null)}>
          <section className="detail-image-modal card" onClick={(event) => event.stopPropagation()}>
            <header className="detail-modal-header">
              <h3>
                {assetViewer.title || (assetViewer.kind === "table" ? "Table" : assetViewer.kind === "document" ? "Document" : "Image")}
              </h3>
              <div className="detail-image-actions">
                {(assetViewer.kind !== "table" && assetViewer.src) && (
                  <a href={assetViewer.src} target="_blank" rel="noreferrer">
                    Open in browser
                  </a>
                )}
                {assetViewer.kind === "image" && (
                  <>
                    <button type="button" onClick={() => adjustImageZoom(-0.25)}>-</button>
                    <button type="button" onClick={resetImageZoom}>100%</button>
                    <button type="button" onClick={() => adjustImageZoom(0.25)}>+</button>
                    <span className="detail-zoom-badge">{Math.round((assetViewer.zoom || 1) * 100)}%</span>
                  </>
                )}
                {(assetViewer.kind === "table" && assetViewer.sourceHref) && (
                  <a href={assetViewer.sourceHref} target="_blank" rel="noreferrer">
                    Open source
                  </a>
                )}
                <button onClick={() => setAssetViewer(null)}>Close</button>
              </div>
            </header>
            {assetViewer.kind === "table" ? (
              <div className="detail-image-stage detail-table-stage">
                <div className="detail-table-preview detail-table-preview-large">
                  <div className="detail-table-caption">
                    Full table view ({assetViewer.tablePreview?.totalRows || 0} rows, {assetViewer.tablePreview?.totalCols || 0} columns)
                  </div>
                  <table>
                    {assetViewer.tablePreview?.columns?.length > 0 && (
                      <thead>
                        <tr>
                          {assetViewer.tablePreview.columns.map((col, colIdx) => (
                            <th key={`viewer-col-${colIdx}`}>{col}</th>
                          ))}
                        </tr>
                      </thead>
                    )}
                    <tbody>
                      {asArray(assetViewer.tablePreview?.rows).map((row, rowIdx) => (
                        <tr key={`viewer-row-${rowIdx}`}>
                          {asArray(row).map((cell, cellIdx) => (
                            <td key={`viewer-cell-${rowIdx}-${cellIdx}`}>{cell}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : assetViewer.kind === "document" ? (
              <div className="detail-image-stage detail-document-stage">
                <iframe
                  className="detail-document-frame"
                  src={assetViewer.src}
                  title={assetViewer.title || "Document"}
                />
              </div>
            ) : (
              <div className="detail-image-stage">
                <img
                  className="detail-image-full"
                  src={assetViewer.src}
                  alt={assetViewer.title || "Image"}
                  style={{
                    width: `${Math.round((assetViewer.zoom || 1) * 100)}%`,
                    maxWidth: "none",
                  }}
                />
              </div>
            )}
          </section>
        </div>
      )}

      <footer className={`event-bar ${eventBar.level}`}>
        <span>{eventBar.text}</span>
        <span>{eventBar.ts}</span>
      </footer>
    </div>
  );
}
