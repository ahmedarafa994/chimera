import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Security rules to prevent API key exposure
  {
    rules: {
      // Warn about console.log in production code
      "no-console": ["warn", { allow: ["warn", "error"] }],
      // Prevent hardcoded secrets
      "no-restricted-syntax": [
        "error",
        {
          selector: "Literal[value=/sk-[a-zA-Z0-9]{20,}/]",
          message: "Potential OpenAI API key detected. Use environment variables instead.",
        },
        {
          selector: "Literal[value=/AIza[a-zA-Z0-9_-]{35}/]",
          message: "Potential Google API key detected. Use environment variables instead.",
        },
        {
          // Private key detection - pattern split to avoid self-triggering
          selector: "Literal[value=/-----BEGIN" + ".*PRIVATE KEY-----/]",
          message: "Private key detected in code. Never commit private keys.",
        },
      ],
      // Prevent accessing non-public env vars client-side
      "no-restricted-properties": [
        "warn",
        {
          object: "process",
          property: "env",
          message: "Access process.env carefully. Only NEXT_PUBLIC_ vars are safe client-side.",
        },
      ],
      // Demote no-explicit-any to warning - too many existing usages to fix immediately
      // These should be addressed incrementally in future refactoring
      "@typescript-eslint/no-explicit-any": "warn",
      // Demote unused vars to warning when prefixed with underscore
      "@typescript-eslint/no-unused-vars": ["warn", {
        argsIgnorePattern: "^_",
        varsIgnorePattern: "^_",
        caughtErrorsIgnorePattern: "^_"
      }],
      // Allow require() in config files (CommonJS compatibility)
      "@typescript-eslint/no-require-imports": "warn",
      // Allow @ts-expect-error with description
      "@typescript-eslint/ban-ts-comment": ["warn", {
        "ts-expect-error": "allow-with-description",
        "ts-ignore": false,
        "ts-nocheck": true,
        "ts-check": false,
        minimumDescriptionLength: 10
      }],
      // Disable React Compiler experimental rules - these are from React 19+ and need significant refactoring
      // The codebase uses patterns that are incompatible with the new compiler
      "react-compiler/react-compiler": "off",
      // Demote React hooks rules to warnings for gradual migration
      "react-hooks/refs": "warn",
      "react-hooks/set-state-in-effect": "warn",
      "react-hooks/incompatible-library": "warn",
      "react-hooks/unsupported-syntax": "warn",
      "react-hooks/immutability": "warn",
      "react-hooks/rules-of-hooks": "warn",
      // Demote Next.js module assignment warnings (used in some optimization patterns)
      "@next/next/no-assign-module-variable": "warn",
      // Demote unused expressions to warning (used in some patterns like void 0)
      "@typescript-eslint/no-unused-expressions": "warn",
      // Demote empty object type (used in some interface patterns)
      "@typescript-eslint/no-empty-object-type": "warn",
    },
  },
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    // Ignore public assets that don't need linting
    "public/**",
    // Ignore scripts directory (CommonJS utilities)
    "scripts/**",
  ]),
]);

export default eslintConfig;

