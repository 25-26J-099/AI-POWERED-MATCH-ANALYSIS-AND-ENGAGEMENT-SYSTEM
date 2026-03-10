import { Radar, RadarChart as RechartsRadar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Legend, Tooltip } from 'recharts';

interface DataPoint {
    name: string;
    values: Record<string, number>;
}

interface Props {
    data: DataPoint[];
}

const COLORS = ['#6366f1', '#a855f7', '#22c55e', '#f59e0b'];

export default function RadarChart({ data }: Props) {
    if (!data.length) return null;

    const keys = Object.keys(data[0].values);
    const chartData = keys.map(key => {
        const point: Record<string, any> = { subject: key };
        data.forEach((d) => {
            point[d.name] = d.values[key];
        });
        return point;
    });

    return (
        <ResponsiveContainer width="100%" height={350}>
            <RechartsRadar cx="50%" cy="50%" outerRadius="70%" data={chartData}>
                <PolarGrid stroke="rgba(99,102,241,0.15)" />
                <PolarAngleAxis
                    dataKey="subject"
                    tick={{ fill: '#94a3b8', fontSize: 12 }}
                />
                <PolarRadiusAxis
                    angle={30}
                    domain={[0, 100]}
                    tick={{ fill: '#64748b', fontSize: 10 }}
                />
                {data.map((d, i) => (
                    <Radar
                        key={d.name}
                        name={d.name}
                        dataKey={d.name}
                        stroke={COLORS[i % COLORS.length]}
                        fill={COLORS[i % COLORS.length]}
                        fillOpacity={0.15}
                        strokeWidth={2}
                    />
                ))}
                <Tooltip
                    contentStyle={{
                        background: '#1a1f35',
                        border: '1px solid rgba(99,102,241,0.3)',
                        borderRadius: '8px',
                        color: '#f1f5f9',
                    }}
                />
                {data.length > 1 && <Legend />}
            </RechartsRadar>
        </ResponsiveContainer>
    );
}
