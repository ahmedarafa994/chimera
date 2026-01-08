import * as React from "react"
import { cn } from "@/lib/utils"

interface GlassCardProps extends React.HTMLAttributes<HTMLDivElement> {
    variant?: "default" | "hover" | "heavy"
    intensity?: "low" | "medium" | "high"
}

const GlassCard = React.forwardRef<HTMLDivElement, GlassCardProps>(
    ({ className, variant = "default", intensity = "medium", ...props }, ref) => {

        const intensityMap = {
            low: "backdrop-blur-sm bg-white/[0.02]",
            medium: "backdrop-blur-md bg-white/[0.05]",
            high: "backdrop-blur-xl bg-white/[0.08]"
        }

        const variantStyles = {
            default: "border border-white/[0.05] shadow-sm",
            hover: "border border-white/[0.05] shadow-sm transition-all duration-300 hover:bg-white/[0.08] hover:border-white/[0.1] hover:shadow-md hover:-translate-y-0.5 cursor-pointer",
            heavy: "border border-white/[0.08] shadow-lg bg-black/40"
        }

        return (
            <div
                ref={ref}
                className={cn(
                    "rounded-xl relative overflow-hidden",
                    intensityMap[intensity],
                    variantStyles[variant],
                    className
                )}
                {...props}
            >
                {/* Subtle noise texture overlay for 'premium' feel */}
                <div className="absolute inset-0 opacity-[0.03] pointer-events-none bg-[url('/noise.png')] mix-blend-overlay" />

                {/* Gradient border effect container */}
                <div className="relative z-10">
                    {props.children}
                </div>
            </div>
        )
    }
)
GlassCard.displayName = "GlassCard"

export { GlassCard }
