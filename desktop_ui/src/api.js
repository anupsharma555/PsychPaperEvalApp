import { invoke } from "@tauri-apps/api/core";

const defaultApiBase =
  window.__PAPEREVAL_API_BASE ||
  import.meta.env.VITE_API_BASE ||
  "http://127.0.0.1:8000/api";

export const API_BASE = defaultApiBase;

export class DesktopApiError extends Error {
  constructor(message, { errorCode = "unknown_error", nextAction = "Retry.", statusCode = 500 } = {}) {
    super(message);
    this.name = "DesktopApiError";
    this.errorCode = errorCode;
    this.nextAction = nextAction;
    this.statusCode = statusCode;
  }
}

async function parseErrorResponse(response) {
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (payload && typeof payload === "object") {
    const detail = payload.detail && typeof payload.detail === "object" ? payload.detail : payload;
    const userMessage =
      detail.user_message ||
      detail.message ||
      (typeof payload.detail === "string" ? payload.detail : "Request failed.");
    const nextAction = detail.next_action || "Retry the request.";
    const errorCode = detail.error_code || `http_${response.status}`;
    return new DesktopApiError(userMessage, {
      errorCode,
      nextAction,
      statusCode: response.status,
    });
  }

  const raw = await response.text().catch(() => "");
  return new DesktopApiError(raw || "Request failed.", {
    errorCode: `http_${response.status}`,
    nextAction: "Retry the request.",
    statusCode: response.status,
  });
}

async function apiRequest(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...(options.headers || {}),
    },
  });
  if (!response.ok) {
    throw await parseErrorResponse(response);
  }
  return response.json();
}

export async function fetchBootstrap() {
  return apiRequest("/desktop/bootstrap");
}

export async function fetchStatus() {
  return apiRequest("/status");
}

export async function fetchJobs({ status, limit = 200, offset = 0, sort = "updated_at:desc" } = {}) {
  const params = new URLSearchParams();
  if (status) params.set("status", status);
  params.set("limit", String(limit));
  params.set("offset", String(offset));
  params.set("sort", sort);
  return apiRequest(`/jobs?${params.toString()}`);
}

export async function fetchRuntimeEvents({ since, limit = 25 } = {}) {
  const params = new URLSearchParams();
  params.set("limit", String(limit));
  if (since) params.set("since", since);
  return apiRequest(`/runtime/events?${params.toString()}`);
}

export async function submitFromUrl(url, doi) {
  return apiRequest("/documents/from-url", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, doi: doi || null, fetch_supplements: true }),
  });
}

export async function submitFromUpload(mainFile, supplements = []) {
  const formData = new FormData();
  formData.append("main_file", mainFile);
  supplements.forEach((file) => formData.append("supp_files", file));
  return apiRequest("/documents/upload", {
    method: "POST",
    body: formData,
  });
}

export async function fetchReportSummary(documentId) {
  return apiRequest(`/documents/${documentId}/report/summary`);
}

export async function fetchReport(documentId) {
  return apiRequest(`/documents/${documentId}/report`);
}

export async function fetchReportSaveStatus(documentId) {
  return apiRequest(`/documents/${documentId}/report/save-status`);
}

export async function saveReport(documentId) {
  return apiRequest(`/documents/${documentId}/report/save`, { method: "POST" });
}

export async function fetchDocumentMedia(documentId) {
  return apiRequest(`/documents/${documentId}/media`);
}

export async function pauseProcessing() {
  return apiRequest("/processing/stop", { method: "POST" });
}

export async function resumeProcessing() {
  return apiRequest("/processing/start", { method: "POST" });
}

export async function cleanOrphans() {
  return apiRequest("/processing/cleanup", { method: "POST" });
}

export async function recoverProcessing() {
  return apiRequest("/processing/recover", { method: "POST" });
}

export async function stopBackend() {
  return apiRequest("/stop", { method: "POST" });
}

export async function openLogsFolder() {
  return invoke("open_logs_folder");
}

export async function getDesktopRuntimeInfo() {
  return invoke("runtime_info");
}

export async function restartBackend() {
  return invoke("restart_backend");
}

export async function exportDiagnosticsBundle() {
  return invoke("export_diagnostics");
}
