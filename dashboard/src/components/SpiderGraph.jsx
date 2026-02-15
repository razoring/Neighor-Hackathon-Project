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
        <div className="bg-white p-6 rounded-3xl shadow-sm border border-gray-100 h-full flex flex-col min-h-[400px]">
            <h3 className="text-xl font-semibold text-gray-800 mb-2">Overall Score Analysis</h3>
            <div className="flex-1 w-full relative">
                <ResponsiveContainer width="100%" height="100%" minHeight={300}>
                    <RadarChart cx="50%" cy="50%" outerRadius="80%" data={data}>
                        <PolarGrid stroke="#e5e7eb" />
                        <PolarAngleAxis dataKey="subject" tick={{ fill: '#6b7280', fontSize: 12 }} />
                        <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                        <Radar
                            name="Patient"
                            dataKey="A"
                            stroke="#8b5cf6"
                            strokeWidth={3}
                            fill="url(#colorGradient)"
                            fillOpacity={0.6}
                        />
                        <defs>
                            <linearGradient id="colorGradient" x1="0" y1="0" x2="1" y2="1">
                                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8} />
                                <stop offset="95%" stopColor="#ec4899" stopOpacity={0.8} />
                            </linearGradient>
                        </defs>
                    </RadarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default SpiderGraph;
