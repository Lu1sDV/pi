import { describe, expect, it } from "vitest";
import { getOpenRouterThinkingLevelMap } from "../catalog/sources/openrouter.ts";
import { streamSimple } from "../src/api/openai-completions.ts";
import type { Model, SimpleStreamOptions, ThinkingLevelMap } from "../src/types.ts";
import { normalizeContext } from "../src/utils/transcript.ts";

const context = normalizeContext({
	messages: [{ role: "user", content: "Hello", timestamp: 0 }],
});

function openRouterModel(thinkingLevelMap?: ThinkingLevelMap): Model<"openai-completions"> {
	return {
		id: "stealth/ox-alpha",
		name: "Ox Alpha",
		api: "openai-completions",
		provider: "openrouter",
		baseUrl: "https://example.invalid/v1",
		reasoning: true,
		thinkingLevelMap,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128000,
		maxTokens: 4096,
		compat: { thinkingFormat: "openrouter" },
	};
}

async function capturePayload(model: Model<"openai-completions">, options: SimpleStreamOptions = {}) {
	let payload: { reasoning?: { effort?: string }; max_tokens?: number; max_completion_tokens?: number } | undefined;
	await streamSimple(model, context, {
		apiKey: "test",
		...options,
		onPayload: (request) => {
			payload = request as NonNullable<typeof payload>;
			throw new Error("payload captured");
		},
	}).result();
	if (!payload) throw new Error("OpenRouter payload was not captured");
	return payload;
}

describe("getOpenRouterThinkingLevelMap", () => {
	it("marks mandatory reasoning and unsupported efforts unavailable", () => {
		expect(
			getOpenRouterThinkingLevelMap({
				mandatory: true,
				default_enabled: true,
				supported_efforts: ["max", "high", "low"],
				default_effort: "max",
			}),
		).toEqual({
			off: null,
			minimal: null,
			low: "low",
			medium: null,
			high: "high",
			xhigh: null,
			max: "max",
		});
	});

	it("still marks off unavailable when OpenRouter omits effort metadata", () => {
		expect(getOpenRouterThinkingLevelMap({ mandatory: true })).toEqual({ off: null });
	});

	it("keeps off available while restricting optional models to supported efforts", () => {
		expect(
			getOpenRouterThinkingLevelMap({
				mandatory: false,
				default_enabled: true,
				supported_efforts: ["high", "low"],
			}),
		).toEqual({
			off: "none",
			minimal: null,
			low: "low",
			medium: null,
			high: "high",
			xhigh: null,
			max: null,
		});
	});

	it("does not add metadata for optional models without effort controls", () => {
		expect(getOpenRouterThinkingLevelMap({ mandatory: false })).toBeUndefined();
	});
});

describe("OpenRouter mandatory reasoning payloads", () => {
	const mandatoryMap = getOpenRouterThinkingLevelMap({
		mandatory: true,
		supported_efforts: ["max", "high", "low"],
	});

	it("omits reasoning when a background call does not request it", async () => {
		expect(await capturePayload(openRouterModel(mandatoryMap))).not.toHaveProperty("reasoning");
	});

	it("still sends an explicitly selected supported effort", async () => {
		expect(await capturePayload(openRouterModel(mandatoryMap), { reasoning: "low" })).toMatchObject({
			reasoning: { effort: "low" },
		});
	});

	it("continues to explicitly disable reasoning for optional models", async () => {
		expect(await capturePayload(openRouterModel())).toMatchObject({
			reasoning: { effort: "none" },
		});
	});
});

describe("OpenRouter output budgets", () => {
	it("does not reserve a context-sized output budget for Kimi by default", async () => {
		const model = {
			...openRouterModel(),
			id: "moonshotai/kimi-k3.1-pro",
			contextWindow: 1_048_576,
			maxTokens: 943_718,
			thinkingLevelMap: { max: "max" },
		};
		expect(await capturePayload(model, { reasoning: "max" })).toMatchObject({
			max_completion_tokens: 32_000,
			reasoning: { effort: "max" },
		});
	});

	it("preserves an explicitly requested larger output budget", async () => {
		const model = { ...openRouterModel(), contextWindow: 1_048_576, maxTokens: 943_718 };
		expect(await capturePayload(model, { maxTokens: 64_000 })).toMatchObject({ max_completion_tokens: 64_000 });
	});

	it("respects a smaller model output limit", async () => {
		expect(await capturePayload(openRouterModel())).toMatchObject({ max_completion_tokens: 4096 });
	});

	it("still leaves room for input and the context safety margin", async () => {
		const model = { ...openRouterModel(), contextWindow: 10_000, maxTokens: 943_718 };
		expect(await capturePayload(model)).toMatchObject({ max_completion_tokens: 5902 });
	});

	it("also caps custom providers using a regional OpenRouter endpoint", async () => {
		const model = {
			...openRouterModel(),
			provider: "custom-openrouter",
			baseUrl: "https://us.openrouter.ai/api/v1",
			contextWindow: 1_048_576,
			maxTokens: 943_718,
			compat: { maxTokensField: "max_tokens" as const },
		};
		expect(await capturePayload(model)).toMatchObject({ max_tokens: 32_000 });
	});

	it("does not impose the OpenRouter credit reservation cap on other providers", async () => {
		const model = {
			...openRouterModel(),
			provider: "openai",
			contextWindow: 1_048_576,
			maxTokens: 128_000,
		};
		expect(await capturePayload(model)).toMatchObject({ max_completion_tokens: 128_000 });
	});
});
