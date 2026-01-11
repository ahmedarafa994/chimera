/**
 * Auth Components Export
 *
 * Re-exports all authentication-related components for convenient imports.
 *
 * @module components/auth
 */

// =============================================================================
// Form Components
// =============================================================================

export { LoginForm } from "./LoginForm";
export type { default as LoginFormDefault } from "./LoginForm";

export { RegisterForm } from "./RegisterForm";
export type { default as RegisterFormDefault } from "./RegisterForm";

export {
  PasswordStrengthMeter,
  PasswordRequirements,
} from "./PasswordStrengthMeter";
export type { default as PasswordStrengthMeterDefault } from "./PasswordStrengthMeter";

export { ForgotPasswordForm } from "./ForgotPasswordForm";
export type { default as ForgotPasswordFormDefault } from "./ForgotPasswordForm";

export { ResetPasswordForm } from "./ResetPasswordForm";
export type { default as ResetPasswordFormDefault } from "./ResetPasswordForm";

// =============================================================================
// Route Protection Components
// =============================================================================

export {
  ProtectedRoute,
  withProtectedRoute,
  useProtectedPage,
} from "./ProtectedRoute";
export type {
  ProtectedRouteProps,
  WithProtectedRouteOptions,
} from "./ProtectedRoute";

export {
  RoleGuard,
  withRoleGuard,
  AccessDenied,
  AdminOnly,
  ResearcherOnly,
} from "./RoleGuard";
export type {
  RoleGuardProps,
  AccessDeniedProps,
  WithRoleGuardOptions,
} from "./RoleGuard";
