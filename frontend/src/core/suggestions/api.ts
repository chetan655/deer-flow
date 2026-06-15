import { fetch } from "@/core/api/fetcher";
import { getBackendBaseURL } from "@/core/config";

export async function loadSuggestionsConfig() {
  const response = await fetch(`${getBackendBaseURL()}/api/suggestions/config`);
  if (!response.ok) {
    // Fallback to true if the backend is older or throws an error
    return { enabled: true };
  }
  return response.json() as Promise<{ enabled: boolean }>;
}
