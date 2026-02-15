import React from 'react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';

const SpiderGraph = ({ dementiaData, breathingData }) => {
    // Map our metrics to a format suitable for the Radar chart.
    // We normalize values to a 0-100 scale for visualization where possible.

    if (!dementiaData || !breathingData) return null;

    const score = dementiaData.score || 0;
    // Inverse irregularity for "Regularity" score (0-1: 0 is regular -> 100, 1 is irregular -> 0)
    // CV is usually small, so we might inverse it.
    const consistency = breathingData.interval_cv ? Math.max(0, 100 - (breathingData.interval_cv * 100)) : 50;

    // BPM: Assuming normal is around 12-20. 
    // Let's just normalize raw BPM to a visual scale (cap at 60).
    const rate = Math.min(100, (breathingData.breath_rate_bpm / 40) * 100);

    // Confidence mapping
    const confidenceMap = { "High": 100, "Medium": 60, "Low": 30 };
    const conf = confidenceMap[dementiaData.confidence] || 50;

    // Laboured (Bool): 100 if Normal, 20 if Laboured
    const effort = breathingData.laboured ? 20 : 100;

    const data = [
        { subject: 'Cognitive Score', A: score, fullMark: 100 },
        { subject: 'Regularity', A: consistency, fullMark: 100 },
        { subject: 'Breathing Rate', A: rate, fullMark: 100 },
        { subject: 'Confidence', A: conf, fullMark: 100 },
        { subject: 'Ease of Breath', A: effort, fullMark: 100 },
    ];

    return (
        <div className="bg-white p-6 rounded-3xl shadow-sm border border-gray-100 h-full flex flex-col">
            <h3 className="text-xl font-semibold text-gray-800 mb-2">Overall Score Analysis</h3>
            <div className="flex-1 w-full min-h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
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
