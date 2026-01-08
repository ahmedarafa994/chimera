"use client"

/**
 * Resizable Panels Component
 *
 * Wrapper around react-resizable-panels for UI consistency.
 * Using a simplified implementation due to version compatibility.
 */

import * as React from "react"
// eslint-disable-next-line @typescript-eslint/no-require-imports
const Resizable = require("react-resizable-panels")
import { GripVertical } from "lucide-react"

import { cn } from "@/lib/utils"

type PanelGroupProps = React.HTMLAttributes<HTMLDivElement> & {
    direction?: "horizontal" | "vertical"
    onLayout?: (sizes: number[]) => void
    autoSaveId?: string
    children?: React.ReactNode
}

type PanelProps = React.HTMLAttributes<HTMLDivElement> & {
    defaultSize?: number
    minSize?: number
    maxSize?: number
    collapsible?: boolean
    onCollapse?: () => void
    onExpand?: () => void
    children?: React.ReactNode
}

type HandleProps = React.HTMLAttributes<HTMLDivElement> & {
    withHandle?: boolean
}

const ResizablePanelGroup = React.forwardRef<HTMLDivElement, PanelGroupProps>(
    ({ className, direction = "horizontal", children, ...props }, ref) => {
        const PanelGroup = Resizable.PanelGroup || Resizable.default?.PanelGroup

        if (!PanelGroup) {
            // Fallback to simple div if PanelGroup not found
            return (
                <div
                    ref={ref}
                    className={cn(
                        "flex h-full w-full",
                        direction === "vertical" ? "flex-col" : "flex-row",
                        className
                    )}
                    {...props}
                >
                    {children}
                </div>
            )
        }

        return (
            <PanelGroup
                ref={ref}
                direction={direction}
                className={cn(
                    "flex h-full w-full",
                    direction === "vertical" ? "flex-col" : "flex-row",
                    className
                )}
                {...props}
            >
                {children}
            </PanelGroup>
        )
    }
)
ResizablePanelGroup.displayName = "ResizablePanelGroup"

const ResizablePanel = React.forwardRef<HTMLDivElement, PanelProps>(
    ({ className, children, ...props }, ref) => {
        const Panel = Resizable.Panel || Resizable.default?.Panel

        if (!Panel) {
            return (
                <div ref={ref} className={cn("flex-1", className)} {...props}>
                    {children}
                </div>
            )
        }

        return (
            <Panel ref={ref} className={className} {...props}>
                {children}
            </Panel>
        )
    }
)
ResizablePanel.displayName = "ResizablePanel"

const ResizableHandle = React.forwardRef<HTMLDivElement, HandleProps>(
    ({ withHandle, className, ...props }, ref) => {
        const PanelResizeHandle = Resizable.PanelResizeHandle || Resizable.default?.PanelResizeHandle

        if (!PanelResizeHandle) {
            return (
                <div
                    ref={ref}
                    className={cn(
                        "relative flex w-px items-center justify-center bg-border",
                        className
                    )}
                    {...props}
                >
                    {withHandle && (
                        <div className="z-10 flex h-4 w-3 items-center justify-center rounded-sm border bg-border">
                            <GripVertical className="h-2.5 w-2.5" />
                        </div>
                    )}
                </div>
            )
        }

        return (
            <PanelResizeHandle
                ref={ref}
                className={cn(
                    "relative flex w-px items-center justify-center bg-border after:absolute after:inset-y-0 after:left-1/2 after:w-1 after:-translate-x-1/2 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-offset-1 data-[panel-group-direction=vertical]:h-px data-[panel-group-direction=vertical]:w-full data-[panel-group-direction=vertical]:after:left-0 data-[panel-group-direction=vertical]:after:h-1 data-[panel-group-direction=vertical]:after:w-full data-[panel-group-direction=vertical]:after:-translate-y-1/2 data-[panel-group-direction=vertical]:after:translate-x-0 [&[data-panel-group-direction=vertical]>div]:rotate-90",
                    className
                )}
                {...props}
            >
                {withHandle && (
                    <div className="z-10 flex h-4 w-3 items-center justify-center rounded-sm border bg-border">
                        <GripVertical className="h-2.5 w-2.5" />
                    </div>
                )}
            </PanelResizeHandle>
        )
    }
)
ResizableHandle.displayName = "ResizableHandle"

export { ResizablePanelGroup, ResizablePanel, ResizableHandle }
