import { USE_MOCK_API } from "./config";
import { restApiClient } from "./endpoints";
import { mockApiClient } from "./mocks/mockClient";
import type { RouteDetectionApiClient } from "./contract";

/**
 * The single entry point components should import from. Resolves to the
 * mock or real backend based on `NEXT_PUBLIC_USE_MOCK_API`
 * (see ./config.ts) -- callers never need to know which one is active.
 */
export const apiClient: RouteDetectionApiClient = USE_MOCK_API
  ? mockApiClient
  : restApiClient;

export * from "./contract";
export * from "./types";
