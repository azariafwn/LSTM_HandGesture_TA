import PrimaryButton from '@/Components/PrimaryButton';
import { Head, Link, useForm } from '@inertiajs/react';

export default function VerifyEmail({ status }) {
    const { post, processing } = useForm({});

    const submit = (e) => {
        e.preventDefault();

        post(route('verification.send'));
    };

    return (
        // Wrapper Utama (Gaya Cyberpunk/Futuristik)
        <div className="min-h-screen flex flex-col sm:justify-center items-center pt-6 sm:pt-0 bg-slate-950 font-mono text-slate-300">
            <Head title="Email Verification" />

            {/* Header */}
            <div className="mb-6 text-center">
                <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 tracking-tighter">
                    IDENTITY_VERIFICATION
                </h1>
                <p className="text-xs text-slate-500 mt-1 tracking-widest">EMAIL_CONFIRMATION_REQUIRED</p>
            </div>

            {/* Card Container */}
            <div className="w-full sm:max-w-md mt-6 px-6 py-8 bg-slate-900/50 border border-slate-800 shadow-[0_0_15px_rgba(0,0,0,0.5)] overflow-hidden sm:rounded-xl backdrop-blur-sm relative group">
                
                {/* Efek Sudut Dekoratif */}
                <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-cyan-500/50 rounded-tl-lg"></div>
                <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-cyan-500/50 rounded-tr-lg"></div>
                <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-cyan-500/50 rounded-bl-lg"></div>
                <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-cyan-500/50 rounded-br-lg"></div>

                <div className="mb-4 text-xs text-slate-400 text-justify leading-relaxed border-l-2 border-cyan-500/30 pl-3">
                    THANKS FOR JOINING THE NETWORK. BEFORE PROCEEDING, PLEASE VERIFY YOUR COMM-LINK (EMAIL) BY CLICKING THE LINK WE JUST SENT. IF YOU DIDN'T RECEIVE IT, WE WILL GLADLY DISPATCH ANOTHER.
                </div>

                {status === 'verification-link-sent' && (
                    <div className="mb-4 text-xs font-medium text-emerald-400 bg-emerald-900/20 p-3 rounded border border-emerald-800 tracking-wide">
                        A NEW VERIFICATION LINK HAS BEEN SENT TO YOUR REGISTERED ADDRESS.
                    </div>
                )}

                <form onSubmit={submit}>
                    <div className="mt-6 flex items-center justify-between">
                        <PrimaryButton 
                            className="bg-cyan-600 hover:bg-cyan-500 text-white font-bold tracking-wider px-4 py-2 rounded-lg shadow-[0_0_10px_rgba(6,182,212,0.3)] transition-all border border-cyan-400 text-xs" 
                            disabled={processing}
                        >
                            {processing ? 'SENDING...' : 'RESEND_LINK'}
                        </PrimaryButton>

                        <Link
                            href={route('logout')}
                            method="post"
                            as="button"
                            className="underline text-xs text-slate-500 hover:text-cyan-400 rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 transition-colors tracking-widest"
                        >
                            TERMINATE_SESSION
                        </Link>
                    </div>
                </form>
            </div>

            {/* Footer */}
            <div className="mt-8 text-[10px] text-slate-600 tracking-widest">
                SECURE SYSTEM V1.0 • PENDING VERIFICATION
            </div>
        </div>
    );
}