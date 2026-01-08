'use client'

import React, { Suspense } from 'react'

interface LazyLoadWrapperProps {
    children: React.ReactNode
    fallback?: React.ReactNode
    className?: string
}

const DefaultSkeleton = () => (
    <div className="w-full h-full min-h-[100px] animate-pulse rounded-lg bg-gray-200 dark:bg-gray-800 flex items-center justify-center">
        <span className="sr-only">Loading...</span>
    </div>
)

export const LazyLoadWrapper = ({
    children,
    fallback,
    className
}: LazyLoadWrapperProps) => {
    return (
        <div className={className}>
            <Suspense fallback={fallback || <DefaultSkeleton />}>
                {children}
            </Suspense>
        </div>
    )
}
