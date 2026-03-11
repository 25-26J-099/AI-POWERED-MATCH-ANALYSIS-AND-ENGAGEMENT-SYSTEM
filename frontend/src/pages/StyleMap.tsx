import { useEffect, useState, useRef } from 'react';
import { useParams } from 'react-router-dom';
import { getStyleMap } from '../api/client';
import * as d3 from 'd3';

export default function StyleMap() {
    const { id } = useParams();
    const [data, setData] = useState<any>(null);
    const [mode, setMode] = useState<'umap' | 'tsne'>('umap');
    const svgRef = useRef<SVGSVGElement>(null);

    useEffect(() => {
        getStyleMap(Number(id)).then(r => setData(r.data)).catch(() => { });
    }, [id]);

    useEffect(() => {
        if (!data || !svgRef.current) return;
        const players = data.players;
        if (!players.length) return;

        const svg = d3.select(svgRef.current);
        svg.selectAll('*').remove();

        const width = 600, height = 500;
        const margin = { top: 30, right: 30, bottom: 30, left: 30 };

        const xKey = mode === 'umap' ? 'umap_x' : 'tsne_x';
        const yKey = mode === 'umap' ? 'umap_y' : 'tsne_y';

        const xExtent = d3.extent(players, (d: any) => d[xKey] as number) as unknown as [number, number];
        const yExtent = d3.extent(players, (d: any) => d[yKey] as number) as unknown as [number, number];

        const xScale = d3.scaleLinear().domain([xExtent[0] - 1, xExtent[1] + 1]).range([margin.left, width - margin.right]);
        const yScale = d3.scaleLinear().domain([yExtent[0] - 1, yExtent[1] + 1]).range([height - margin.bottom, margin.top]);

        const clusters = [...new Set(players.map((p: any) => p.cluster))];
        const colorScale = d3.scaleOrdinal(d3.schemeSet2).domain(clusters.map(String));

        // Draw dots
        const dots = svg.selectAll('circle')
            .data(players)
            .enter().append('circle')
            .attr('cx', (d: any) => xScale(d[xKey]))
            .attr('cy', (d: any) => yScale(d[yKey]))
            .attr('r', 0)
            .attr('fill', (d: any) => colorScale(String(d.cluster)))
            .attr('stroke', 'rgba(255,255,255,0.3)')
            .attr('stroke-width', 1.5)
            .style('cursor', 'pointer');

        dots.transition().duration(800).delay((_, i) => i * 30).attr('r', 8);

        // Labels
        svg.selectAll('text.label')
            .data(players)
            .enter().append('text')
            .attr('class', 'label')
            .attr('x', (d: any) => xScale(d[xKey]))
            .attr('y', (d: any) => yScale(d[yKey]) - 14)
            .attr('text-anchor', 'middle')
            .attr('fill', '#94a3b8')
            .attr('font-size', '11px')
            .text((d: any) => d.name?.split(' ').pop() || '');

        // Tooltips
        dots.append('title')
            .text((d: any) => `${d.name} (${d.team})\n${d.cluster_label || `Cluster ${d.cluster}`}`);

    }, [data, mode]);

    return (
        <div className="page-container" style={{ maxWidth: '800px', margin: '0 auto' }}>
            <h1 className="page-title">Player Style Map</h1>
            <p className="page-subtitle">Player style embeddings reduced to 2D — similar players cluster together</p>

            <div style={{ display: 'flex', gap: '8px', marginBottom: '24px' }}>
                {(['umap', 'tsne'] as const).map(m => (
                    <button key={m} onClick={() => setMode(m)} style={{
                        padding: '8px 20px', borderRadius: '8px', fontWeight: 600, textTransform: 'uppercase',
                        background: mode === m ? 'var(--accent)' : 'var(--bg-card)',
                        color: mode === m ? 'white' : 'var(--text-secondary)',
                        border: '1px solid ' + (mode === m ? 'var(--accent)' : 'var(--border-subtle)'),
                        cursor: 'pointer', transition: 'all 0.2s',
                    }}>
                        {m}
                    </button>
                ))}
            </div>

            <div className="glass-card" style={{ padding: '16px' }}>
                {!data ? (
                    <div style={{ textAlign: 'center', padding: '80px' }}><div className="spinner" style={{ margin: '0 auto' }} /></div>
                ) : (
                    <svg ref={svgRef} width="100%" viewBox="0 0 600 500" style={{ maxWidth: '100%' }} />
                )}
            </div>
        </div>
    );
}
