import React from 'react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';

const SpiderGraph = ({ dementiaData, breathingData }) => {
    // Defensive defaults to ensure graph always renders something
    const dData = dementiaData || {};
    const bData = breathingData || {};

    // Score
    const score = typeof dData.score === 'number' ? dData.score : 0;

    // Consistency: Handle null/undefined safely
    // interval_cv might be null if no breaths detected
    let consistency = 50;
    if (typeof bData.interval_cv === 'number') {
        consistency = Math.max(0, 100 - (bData.interval_cv * 100));
    }

    // BPM
    const bpm = typeof bData.breath_rate_bpm === 'number' ? bData.breath_rate_bpm : 0;
    const rate = Math.min(100, (bpm / 40) * 100);

    // Confidence mapping
    const confidenceMap = { "High": 100, "Medium": 60, "Low": 30 };
    const conf = confidenceMap[dData.confidence] || 50;

    // Laboured
    const effort = bData.laboured ? 20 : 100;

    const data = [
        { subject: 'Cognitive Score', A: score, fullMark: 100 },
        { subject: 'Regularity', A: consistency, fullMark: 100 },
        { subject: 'Breathing Rate', A: rate, fullMark: 100 },
        { subject: 'Confidence', A: conf, fullMark: 100 },
        { subject: 'Ease of Breath', A: effort, fullMark: 100 },
    ];

    return (
        <div className="w-full h-full flex flex-col items-center justify-center">
            <ResponsiveContainer width="100%" height="100%" minHeight={300}>
                <RadarChart cx="50%" cy="50%" outerRadius="75%" data={data}>
                    <PolarGrid stroke="#e5e7eb" strokeOpacity={0.5} />
                    <PolarAngleAxis dataKey="subject" tick={{ fill: '#6b7280', fontSize: 11, fontWeight: 500 }} />
                    <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                    <Radar
                        name="Patient"
                        dataKey="A"
                        stroke="#a855f7"
                        strokeWidth={3}
                        fill="url(#colorGradient)"
                        fillOpacity={0.5}
                    />
                    <defs>
                        <linearGradient id="colorGradient" x1="0" y1="0" x2="1" y2="1">
                            <stop offset="5%" stopColor="#a855f7" stopOpacity={0.8} />
                            <stop offset="95%" stopColor="#ec4899" stopOpacity={0.8} />
                        </linearGradient>
                    </defs>
                </RadarChart>
            </ResponsiveContainer>
        </div>
    );
};

export default SpiderGraph;
