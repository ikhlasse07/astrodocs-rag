const API_BASE_URL = (
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function readErrorMessage(data, status) {
  if (typeof data?.detail === "string") {
    return data.detail;
  }

  if (Array.isArray(data?.detail)) {
    return data.detail
      .map((item) => item.msg || "Invalid request.")
      .join(" ");
  }

  if (status === 503) {
    return "The library is busy. Please try again shortly.";
  }

  return "The request could not be completed.";
}

async function request(
  path,
  { body, signal, timeoutMs = 10000 } = {},
) {
  const controller = new AbortController();
  let timedOut = false;

  const forwardAbort = () => controller.abort();

  if (signal?.aborted) {
    controller.abort();
  } else {
    signal?.addEventListener("abort", forwardAbort, { once: true });
  }

  const timeoutId = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      method: body === undefined ? "GET" : "POST",
      headers: {
        Accept: "application/json",
        ...(body === undefined
          ? {}
          : { "Content-Type": "application/json" }),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    });

    const data = await response.json().catch(() => null);

    if (!response.ok) {
      throw new ApiError(
        readErrorMessage(data, response.status),
        response.status,
      );
    }

    if (data === null) {
      throw new ApiError(
        "The server returned an unreadable response.",
        500,
      );
    }

    return data;
  } catch (error) {
    if (signal?.aborted) {
      throw error;
    }

    if (timedOut) {
      throw new ApiError(
        "The request took too long. The server may still be processing it. Wait a moment before trying again.",
        408,
      );
    }

    if (error instanceof ApiError) {
      throw error;
    }

    throw new ApiError(
      "Cannot reach the library. Make sure the backend is running, then check the connection.",
      0,
    );
  } finally {
    window.clearTimeout(timeoutId);
    signal?.removeEventListener("abort", forwardAbort);
  }
}

export async function getHealth(signal) {
  const data = await request("/health", { signal });

  if (data.status !== "ok") {
    throw new ApiError("The library is not ready yet.", 503);
  }

  return data;
}

export async function askQuestion(question, signal) {
  const data = await request("/ask", {
    body: {
      question,
      top_k: 4,
      candidate_k: 20,
    },
    signal,
    timeoutMs: 180000,
  });

  if (
    typeof data.answer !== "string" ||
    !Array.isArray(data.cited_sources)
  ) {
    throw new ApiError(
      "The server returned an unexpected answer format.",
      500,
    );
  }

  return data;
}