import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ChevronDown, ChevronUp } from "lucide-react"
import { useState } from "react"
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
  PaginationEllipsis,
} from "@/components/ui/pagination"
import { RowsPerPageSelect } from "./RowsPerPageSelect"

interface CsvRow {
  id: number;
  text: string;
  llm_label?: string;
  model_label?: string;
  human_label?: string;
  final_label?: string;
}

interface ExpandedRows {
  [key: number]: boolean;
}

interface DataTableProps {
  data: CsvRow[];
  onHover: (index: number | null) => void;
  classes?: string[];
  onLabelChange?: (rowId: number, label: string) => void;
  page: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  pageSize: number;
  onPageSizeChange: (size: number) => void;
}

export function DataTable({ 
  data, 
  onHover, 
  classes = [], 
  onLabelChange,
  page,
  totalPages,
  onPageChange,
  pageSize,
  onPageSizeChange
}: DataTableProps) {
  const [expandedRows, setExpandedRows] = useState<ExpandedRows>({});

  const toggleRowExpansion = (index: number) => {
    setExpandedRows(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  const renderPaginationItems = () => {
    const items = [];
    const maxVisiblePages = 5;
    const halfVisible = Math.floor(maxVisiblePages / 2);

    // Always show first page
    items.push(
      <PaginationItem key={1}>
        <PaginationLink
          onClick={() => onPageChange(1)}
          isActive={page === 1}
          className="cursor-pointer"
        >
          1
        </PaginationLink>
      </PaginationItem>
    );

    let startPage = Math.max(2, page - halfVisible);
    let endPage = Math.min(totalPages - 1, page + halfVisible);

    // Add ellipsis after first page if needed
    if (startPage > 2) {
      items.push(
        <PaginationItem key="start-ellipsis">
          <PaginationEllipsis />
        </PaginationItem>
      );
    }

    // Add middle pages
    for (let i = startPage; i <= endPage; i++) {
      items.push(
        <PaginationItem key={i}>
          <PaginationLink
            onClick={() => onPageChange(i)}
            isActive={page === i}
            className="cursor-pointer"
          >
            {i}
          </PaginationLink>
        </PaginationItem>
      );
    }

    // Add ellipsis before last page if needed
    if (endPage < totalPages - 1) {
      items.push(
        <PaginationItem key="end-ellipsis">
          <PaginationEllipsis />
        </PaginationItem>
      );
    }

    // Always show last page if there is more than one page
    if (totalPages > 1) {
      items.push(
        <PaginationItem key={totalPages}>
          <PaginationLink
            onClick={() => onPageChange(totalPages)}
            isActive={page === totalPages}
            className="cursor-pointer"
          >
            {totalPages}
          </PaginationLink>
        </PaginationItem>
      );
    }

    return items;
  };

  return (
    <Card className="bg-card p-4 h-[calc(100vh-12rem)]">
      <div className="flex flex-col h-full">
        <div className="flex-grow overflow-y-auto space-y-4 mb-4">
        <div className="space-y-2">
          {data.length > 0 ? (
            data.map((row, index) => (
              <div
                key={index}
                className="p-4 rounded-lg border hover:bg-accent transition-colors"
                onMouseEnter={() => onHover(index)}
                // onMouseLeave={() => onHover(null)}
              >
                <div className="flex flex-col gap-2">
                <div className="flex items-start gap-2">
                  <p className={`text-sm flex-grow ${expandedRows[index] ? '' : 'line-clamp-1'}`}>
                    {row.text}
                  </p>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="shrink-0"
                    onClick={() => toggleRowExpansion(index)}
                  >
                    {expandedRows[index] ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                  </Button>
                </div>
                  <div className="flex items-center gap-2">
                    {row.human_label ? (
                      <Badge variant="default">{row.human_label}</Badge>
                    ) : (
                      <Badge variant="destructive">Unlabeled</Badge>
                    )}
                    {classes.length > 0 && onLabelChange && (
                      <div className="flex gap-1">
                        {classes
                          .filter(label => label !== row.human_label)
                          .map((label) => (
                            <Button
                              key={label}
                              variant="outline"
                              size="sm"
                              onClick={() => onLabelChange(row.id, label)}
                            >
                              {label}
                            </Button>
                          ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))
          ) : (
            <p className="text-muted-foreground">
              No CSV data loaded. Please upload and process a file.
            </p>
          )}
        </div>

        </div>
        {data.length > 0 && (
          <div className="flex items-center justify-between border-t pt-4 bg-card">
            <RowsPerPageSelect
              value={pageSize}
              onChange={onPageSizeChange}
            />
            <Pagination>
              <PaginationContent>
                <PaginationItem>
                  <PaginationPrevious 
                    onClick={() => onPageChange(page - 1)}
                    className={page <= 1 ? "pointer-events-none opacity-50" : "cursor-pointer"}
                  />
                </PaginationItem>
                {renderPaginationItems()}
                <PaginationItem>
                  <PaginationNext 
                    onClick={() => onPageChange(page + 1)}
                    className={page >= totalPages ? "pointer-events-none opacity-50" : "cursor-pointer"}
                  />
                </PaginationItem>
              </PaginationContent>
            </Pagination>
          </div>
        )}
      </div>
    </Card>
  )
}
