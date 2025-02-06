"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

interface RowsPerPageSelectProps {
  value: number;
  onChange: (value: number) => void;
  options?: number[];
}

export function RowsPerPageSelect({ 
  value, 
  onChange, 
  options = [10, 20, 50, 100] 
}: RowsPerPageSelectProps) {
  const handleChange = (value: string) => {
    onChange(parseInt(value, 10));
  };

  return (
    <div className="flex items-center gap-2">
      <span className="text-sm text-muted-foreground">Rows per page:</span>
      <Select onValueChange={handleChange} value={value.toString()}>
        <SelectTrigger className="w-[80px]">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((option) => (
            <SelectItem key={option} value={option.toString()}>
              {option}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}
