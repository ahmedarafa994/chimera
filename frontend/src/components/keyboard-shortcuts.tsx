"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Command } from "lucide-react";

export function KeyboardShortcuts() {
    const [isOpen, setIsOpen] = useState(false);
    const router = useRouter();

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            // Toggle help with '?' (Shift + /)
            if (e.key === "?" && !e.ctrlKey && !e.metaKey && !e.altKey && (e.target as HTMLElement).tagName !== "INPUT" && (e.target as HTMLElement).tagName !== "TEXTAREA") {
                e.preventDefault();
                setIsOpen((prev) => !prev);
            }

            // Quick navigation with Ctrl+K (placeholder for command palette, using for search/shortcuts for now)
            if ((e.ctrlKey || e.metaKey) && e.key === "k") {
                e.preventDefault();
                // For now, toggle shortcuts dialog. In future, open command palette.
                setIsOpen((prev) => !prev);
            }

            // Navigation shortcuts
            if ((e.ctrlKey || e.metaKey) && e.key === "1") router.push("/dashboard");
            if ((e.ctrlKey || e.metaKey) && e.key === "2") router.push("/dashboard/jailbreak");
            if ((e.ctrlKey || e.metaKey) && e.key === "3") router.push("/dashboard/autodan");
        };

        window.addEventListener("keydown", handleKeyDown);
        return () => window.removeEventListener("keydown", handleKeyDown);
    }, [router]);

    return (
        <Dialog open={isOpen} onOpenChange={setIsOpen}>
            <DialogContent className="sm:max-w-[425px]">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                        <Command className="h-5 w-5" />
                        Keyboard Shortcuts
                    </DialogTitle>
                    <DialogDescription>
                        Quickly navigate and control Chimera.
                    </DialogDescription>
                </DialogHeader>
                <Table>
                    <TableHeader>
                        <TableRow>
                            <TableHead>Key</TableHead>
                            <TableHead>Action</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                        <TableRow>
                            <TableCell><kbd className="pointer-events-none inline-flex h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium opacity-100">?</kbd></TableCell>
                            <TableCell>Show this help</TableCell>
                        </TableRow>
                        <TableRow>
                            <TableCell><kbd className="pointer-events-none inline-flex h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium opacity-100">Ctrl+K</kbd></TableCell>
                            <TableCell>Toggle Shortcuts / Command</TableCell>
                        </TableRow>
                        <TableRow>
                            <TableCell><kbd className="pointer-events-none inline-flex h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium opacity-100">Ctrl+1</kbd></TableCell>
                            <TableCell>Go to Dashboard</TableCell>
                        </TableRow>
                        <TableRow>
                            <TableCell><kbd className="pointer-events-none inline-flex h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium opacity-100">Ctrl+2</kbd></TableCell>
                            <TableCell>Go to Jailbreak</TableCell>
                        </TableRow>
                        <TableRow>
                            <TableCell><kbd className="pointer-events-none inline-flex h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium opacity-100">Ctrl+3</kbd></TableCell>
                            <TableCell>Go to AutoDAN</TableCell>
                        </TableRow>
                    </TableBody>
                </Table>
            </DialogContent>
        </Dialog>
    );
}
