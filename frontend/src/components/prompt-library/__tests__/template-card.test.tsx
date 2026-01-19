import { render, screen } from "@testing-library/react";
import { PromptTemplateCard } from "../template-card";
import { 
    TemplateListItem, 
    TechniqueType, 
    VulnerabilityType, 
    SharingLevel, 
    TemplateStatus 
} from "@/types/prompt-library-types";
import { describe, it, expect, vi } from "vitest";

// Mock TooltipProvider since it needs a context
vi.mock("@/components/ui/tooltip", () => ({
    Tooltip: ({ children }: any) => <div>{children}</div>,
    TooltipContent: ({ children }: any) => <div>{children}</div>,
    TooltipProvider: ({ children }: any) => <div>{children}</div>,
    TooltipTrigger: ({ children }: any) => <div>{children}</div>,
}));

const mockTemplate: TemplateListItem = {
    id: "1",
    title: "Test Template",
    description: "A test description",
    technique_types: [TechniqueType.AUTODAN],
    vulnerability_types: [VulnerabilityType.JAILBREAK],
    sharing_level: SharingLevel.PUBLIC,
    status: TemplateStatus.ACTIVE,
    avg_rating: 4.5,
    total_ratings: 10,
    effectiveness_score: 0.8,
    tags: ["test", "safe"],
    created_at: new Date().toISOString(),
    owner_id: "user-1"
};

describe("PromptTemplateCard", () => {
    it("renders the template title and description", () => {
        render(<PromptTemplateCard template={mockTemplate} />);
        expect(screen.getByText("Test Template")).toBeDefined();
        expect(screen.getByText("A test description")).toBeDefined();
    });

    it("displays the correct rating", () => {
        render(<PromptTemplateCard template={mockTemplate} />);
        expect(screen.getByText("4.5")).toBeDefined();
        expect(screen.getByText("(10)")).toBeDefined();
    });

    it("displays effectiveness percentage", () => {
        render(<PromptTemplateCard template={mockTemplate} />);
        expect(screen.getByText("80% Effective")).toBeDefined();
    });
});
