import AuthenticatedLayout from '@/Layouts/AuthenticatedLayout';
import { Head } from '@inertiajs/react';
import { useState } from 'react';

export default function Dashboard({ auth, logs }) {
    // Ganti IP ini dengan IP Raspberry Pi kamu!
    // Supaya browser laptop bisa mengambil gambar dari Raspi.
    const [streamUrl, setStreamUrl] = useState('http://100.89.95.109:5000/video_feed');

    const handleRefresh = () => {
        window.location.reload();
    };

    return (
        <AuthenticatedLayout
            user={auth.user}
            header={<h2 className="font-semibold text-xl text-gray-800 leading-tight">Smart Home Dashboard</h2>}
        >
            <Head title="Dashboard" />

            <div className="py-12">
                <div className="max-w-7xl mx-auto sm:px-6 lg:px-8">
                    
                    {/* LAYOUT ATAS: KAMERA & STATISTIK */}
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
                        
                        {/* 1. LIVE CAMERA FEED */}
                        <div className="lg:col-span-2 bg-black rounded-lg overflow-hidden shadow-lg relative">
                            <div className="absolute top-2 left-2 bg-red-600 text-white text-xs px-2 py-1 rounded animate-pulse">
                                LIVE ●
                            </div>
                            {/* Ini tag sakti untuk menampilkan Stream dari Flask */}
                            <img 
                                src={streamUrl} 
                                alt="Live Stream Offline" 
                                className="w-full h-full object-contain bg-gray-900"
                                onError={(e) => {
                                    e.target.style.display = 'none'; 
                                    // Tampilkan pesan error kalau stream mati
                                    e.target.nextSibling.style.display = 'flex';
                                }}
                            />
                            <div className="hidden absolute inset-0 flex-col items-center justify-center text-gray-500">
                                <svg className="w-12 h-12 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636"></path></svg>
                                <span>Kamera Offline / IP Salah</span>
                                <input 
                                    type="text" 
                                    className="mt-2 text-xs text-black p-1 rounded" 
                                    placeholder="Masukkan IP Raspi:5000"
                                    onChange={(e) => setStreamUrl(`http://${e.target.value}/video_feed`)}
                                />
                            </div>
                        </div>

                        {/* 2. STATISTIK KANAN */}
                        <div className="flex flex-col gap-4">
                             <div className="bg-white overflow-hidden shadow-sm sm:rounded-lg p-6">
                                <div className="text-gray-500 text-sm">Status Sistem</div>
                                <div className="text-2xl font-bold text-green-600">Active ⚡</div>
                            </div>
                            <div className="bg-white overflow-hidden shadow-sm sm:rounded-lg p-6">
                                <div className="text-gray-500 text-sm">Gestur Terdeteksi</div>
                                <div className="text-2xl font-bold">{logs.length > 0 ? logs[0].target_gesture : '-'}</div>
                            </div>
                            <div className="bg-white overflow-hidden shadow-sm sm:rounded-lg p-6 flex-grow flex flex-col justify-center items-center">
                                <div className="text-gray-500 text-sm mb-2">Kontrol Data</div>
                                <button 
                                    onClick={handleRefresh}
                                    className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-full transition w-full shadow-lg"
                                >
                                    Refresh Log 🔄
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* LAYOUT BAWAH: TABEL LOG */}
                    <div className="bg-white overflow-hidden shadow-sm sm:rounded-lg">
                        <div className="p-6 text-gray-900">
                            <h3 className="text-lg font-bold mb-4">Riwayat Aktivitas</h3>
                            <div className="overflow-x-auto">
                                <table className="min-w-full table-auto text-sm">
                                    <thead className="bg-gray-100 text-gray-600 uppercase text-xs leading-normal">
                                        <tr>
                                            <th className="py-3 px-6 text-left">Waktu</th>
                                            <th className="py-3 px-6 text-left">Gestur</th>
                                            <th className="py-3 px-6 text-left">Perintah</th>
                                            <th className="py-3 px-6 text-center">Latency</th>
                                            <th className="py-3 px-6 text-center">Status</th>
                                        </tr>
                                    </thead>
                                    <tbody className="text-gray-600 text-sm font-light">
                                        {logs.map((log, index) => (
                                            <tr key={index} className={`border-b border-gray-200 hover:bg-gray-100 ${index === 0 ? 'bg-blue-50' : ''}`}>
                                                <td className="py-3 px-6 text-left whitespace-nowrap font-mono">{log.timestamp}</td>
                                                <td className="py-3 px-6 text-left font-bold text-blue-500">{log.target_gesture}</td>
                                                <td className="py-3 px-6 text-left">
                                                    <span className={`px-2 py-1 rounded-full text-xs ${log.last_command.includes('ON') ? 'bg-green-200 text-green-600' : 'bg-red-200 text-red-600'}`}>
                                                        {log.last_command}
                                                    </span>
                                                </td>
                                                <td className="py-3 px-6 text-center text-red-500 font-mono">{parseFloat(log.wifi_latency_ms).toFixed(0)} ms</td>
                                                <td className="py-3 px-6 text-center">
                                                    {log.last_command.includes('ON') ? '💡' : '🌑'}
                                                </td>
                                            </tr>
                                        ))}
                                        {logs.length === 0 && (
                                            <tr><td colSpan="5" className="text-center py-4">Menunggu data dari Raspberry Pi...</td></tr>
                                        )}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>

                </div>
            </div>
        </AuthenticatedLayout>
    );
}