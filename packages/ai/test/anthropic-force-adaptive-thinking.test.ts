import { describe, expect, it } from "vitest";
import { getModel, streamSimple } from "../src/compat.ts";
import type { Context, Model, SimpleStreamOptions } from "../src/types.ts";

interface AnthropicThinkingPayload {
	max_tokens: number;
	thinking?: { type: string; budget_tokens?: number; display?: string };
	output_config?: { effort?: string };
}

class PayloadCaptured extends Error {
	constructor() {
		super("payload captured");
		this.name = "PayloadCaptured";
	}
}

function makeContext(): Context {
	return {
		messages: [{ role: "user", content: "Hello", timestamp: Date.now() }],
	};
}

function makeCustomModel(compat?: Model<"anthropic-messages">["compat"]): Model<"anthropic-messages"> {
	return {
		// Id intentionally does not match any built-in adaptive substring. This
		// mirrors corporate proxy schemes such as `anthropic--claude-opus-latest`.
		id: "vendor--claude-opus-latest",
		name: "Vendor Proxy Opus Latest",
		api: "anthropic-messages",
		provider: "vendor-proxy",
		baseUrl: "http://127.0.0.1:9",
		reasoning: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 200000,
		maxTokens: 32000,
		compat,
	};
}

async function capturePayload(
	model: Model<"anthropic-messages">,
	options?: SimpleStreamOptions,
): Promise<AnthropicThinkingPayload> {
	let capturedPayload: AnthropicThinkingPayload | undefined;

	const payloadCaptureModel: Model<"anthropic-messages"> = {
		...model,
		baseUrl: "http://127.0.0.1:9",
	};

	const s = streamSimple(payloadCaptureModel, makeContext(), {
		...options,
		apiKey: "fake-key",
		onPayload: (payload) => {
			capturedPayload = payload as AnthropicThinkingPayload;
			throw new PayloadCaptured();
		},
	});

	await s.result();

	if (!capturedPayload) {
		throw new Error("Expected payload to be captured before request failure");
	}

	return capturedPayload;
}

describe("Anthropic forceAdaptiveThinking compat override", () => {
	it("sends legacy thinking payload for custom model ids by default", async () => {
		const payload = await capturePayload(makeCustomModel(), { reasoning: "medium" });

		expect(payload.thinking?.type).toBe("enabled");
		expect(payload.output_config).toBeUndefined();
	});

	it("sends adaptive thinking payload when compat.forceAdaptiveThinking is true", async () => {
		const payload = await capturePayload(makeCustomModel({ forceAdaptiveThinking: true }), { reasoning: "medium" });

		expect(payload.thinking).toEqual({ type: "adaptive", display: "summarized" });
		expect(payload.output_config).toEqual({ effort: "medium" });
	});

	it("uses adaptive thinking with native xhigh effort for Claude Fable 5", async () => {
		const payload = await capturePayload(getModel("anthropic", "claude-fable-5"), { reasoning: "xhigh" });

		expect(payload.thinking).toEqual({ type: "adaptive", display: "summarized" });
		expect(payload.output_config).toEqual({ effort: "xhigh" });
	});

	it("allows built-in adaptive models to opt out with compat.forceAdaptiveThinking false", async () => {
		const model: Model<"anthropic-messages"> = {
			...getModel("anthropic", "claude-opus-4-8"),
			compat: { forceAdaptiveThinking: false },
		};
		const payload = await capturePayload(model, { reasoning: "medium" });

		expect(payload.thinking?.type).toBe("enabled");
		expect(payload.output_config).toBeUndefined();
	});

	it("preserves thinking.type=disabled when reasoning is off regardless of override", async () => {
		const payload = await capturePayload(makeCustomModel({ forceAdaptiveThinking: true }));

		expect(payload.thinking).toEqual({ type: "disabled" });
		expect(payload.output_config).toBeUndefined();
	});
});

describe("OpenRouter Anthropic output budgets", () => {
	const model = { ...makeCustomModel(), provider: "openrouter", maxTokens: 64_000 };

	it("does not reserve the full model capacity when thinking is disabled", async () => {
		expect(await capturePayload(model)).toMatchObject({
			max_tokens: 32_000,
			thinking: { type: "disabled" },
		});
	});

	it("adds budget-based thinking to the capped base output budget", async () => {
		expect(await capturePayload(model, { reasoning: "medium" })).toMatchObject({
			max_tokens: 40_192,
			thinking: { type: "enabled", budget_tokens: 8192 },
		});
	});

	it("keeps adaptive thinking within the capped response budget", async () => {
		const adaptiveModel = { ...model, compat: { forceAdaptiveThinking: true } };
		expect(await capturePayload(adaptiveModel, { reasoning: "medium" })).toMatchObject({
			max_tokens: 32_000,
			thinking: { type: "adaptive" },
			output_config: { effort: "medium" },
		});
	});
});
