import { useRef, useEffect } from 'react';

interface Props {
    data: {
        grid: number[][];
        bins_x: number;
        bins_y: number;
        max_value: number;
    } | null;
}

export default function HeatmapCanvas({ data }: Props) {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        if (!data || !canvasRef.current) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const width = canvas.width;
        const height = canvas.height;
        const { grid, bins_x, bins_y, max_value } = data;

        const cellW = width / bins_x;
        const cellH = height / bins_y;

        // Draw pitch background
        ctx.fillStyle = '#1a5c2e';
        ctx.fillRect(0, 0, width, height);

        // Pitch markings
        ctx.strokeStyle = 'rgba(255,255,255,0.15)';
        ctx.lineWidth = 1;
        ctx.strokeRect(0, 0, width, height);
        // Center line
        ctx.beginPath();
        ctx.moveTo(width / 2, 0);
        ctx.lineTo(width / 2, height);
        ctx.stroke();
        // Center circle
        ctx.beginPath();
        ctx.arc(width / 2, height / 2, 40, 0, Math.PI * 2);
        ctx.stroke();

        // Draw heatmap
        for (let r = 0; r < bins_y; r++) {
            for (let c = 0; c < bins_x; c++) {
                const val = grid[r]?.[c] || 0;
                if (val === 0) continue;

                const intensity = val / max_value;
                const alpha = Math.min(intensity * 0.85, 0.85);

                // Color gradient: blue → yellow → red
                let red, green, blue;
                if (intensity < 0.5) {
                    const t = intensity * 2;
                    red = Math.floor(t * 255);
                    green = Math.floor(t * 200);
                    blue = Math.floor((1 - t) * 200 + 55);
                } else {
                    const t = (intensity - 0.5) * 2;
                    red = 255;
                    green = Math.floor((1 - t) * 200);
                    blue = Math.floor((1 - t) * 55);
                }

                ctx.fillStyle = `rgba(${red},${green},${blue},${alpha})`;

                // Rounded cells for smoother look
                const x = c * cellW;
                const y = r * cellH;
                ctx.beginPath();
                ctx.roundRect(x + 1, y + 1, cellW - 2, cellH - 2, 3);
                ctx.fill();
            }
        }
    }, [data]);

    if (!data) {
        return (
            <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
                No heatmap data available
            </div>
        );
    }

    return (
        <canvas
            ref={canvasRef}
            width={480}
            height={320}
            style={{
                width: '100%',
                borderRadius: '8px',
                border: '2px solid #2d6b3f',
            }}
        />
    );
}
