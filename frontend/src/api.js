const API_BASE = import.meta.env.VITE_API_BASE || "";

export async function fetchProfiles() {
  const response = await fetch(`${API_BASE}/api/profiles`);
  if (!response.ok) throw new Error("Failed to fetch profiles");
  return response.json();
}

function profileFormData(payload) {
  const form = new FormData();
  form.append("profile_name", payload.profile_name || "Untitled profile");
  form.append("selected_topics", JSON.stringify(payload.selected_topics || []));
  form.append("keywords", payload.keywords || "");
  form.append("research_statement", payload.research_statement || "");
  for (const file of payload.seed_papers || []) {
    form.append("seed_papers", file);
  }
  return form;
}

export async function previewProfile(payload) {
  const response = await fetch(`${API_BASE}/api/profiles/preview`, {
    method: "POST",
    body: profileFormData(payload),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "Failed to preview profile");
  }
  return response.json();
}

export async function saveProfile(payload) {
  const response = await fetch(`${API_BASE}/api/profiles/save`, {
    method: "POST",
    body: profileFormData(payload),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "Failed to save profile");
  }
  return response.json();
}

export async function analyzePaper({ profileName, density, profileId, file }) {
  const form = new FormData();
  form.append("profile_name", profileName);
  form.append("density", density);
  form.append("profile_id", profileId || "");
  if (file) form.append("paper", file);

  const response = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    const data = await response.json().catch(() => null);
    throw new Error(data?.error || "Analyze request failed");
  }

  return response.json();
}
