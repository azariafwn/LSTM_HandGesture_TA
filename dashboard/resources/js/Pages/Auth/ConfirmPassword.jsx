import InputError from '@/Components/InputError';
import InputLabel from '@/Components/InputLabel';
import PrimaryButton from '@/Components/PrimaryButton';
import TextInput from '@/Components/TextInput';
import { Head, useForm } from '@inertiajs/react';

export default function ConfirmPassword() {
    const { data, setData, post, processing, errors, reset } = useForm({
        password: '',
    });

    const submit = (e) => {
        e.preventDefault();

        post(route('password.confirm'), {
            onFinish: () => reset('password'),
        });
    };

    return (
        // Wrapper Utama
        <div className="min-h-screen flex flex-col sm:justify-center items-center pt-6 sm:pt-0 bg-slate-950 font-mono text-slate-300">
            <Head title="Confirm Password" />

            {/* Header */}
            <div className="mb-6 text-center">
                <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 tracking-tighter">
                    SECURITY_CHECK
                </h1>
                <p className="text-xs text-slate-500 mt-1 tracking-widest">IDENTITY VERIFICATION REQUIRED</p>
            </div>

            {/* Card Container */}
            <div className="w-full sm:max-w-md mt-6 px-6 py-8 bg-slate-900/50 border border-slate-800 shadow-[0_0_15px_rgba(0,0,0,0.5)] overflow-hidden sm:rounded-xl backdrop-blur-sm relative group">
                
                {/* Dekorasi Sudut */}
                <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-cyan-500/50 rounded-tl-lg"></div>
                <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-cyan-500/50 rounded-tr-lg"></div>
                <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-cyan-500/50 rounded-bl-lg"></div>
                <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-cyan-500/50 rounded-br-lg"></div>

                <div className="mb-4 text-xs text-slate-400 tracking-wide text-justify border-l-2 border-cyan-500/30 pl-3">
                    SECURE AREA DETECTED. PLEASE INPUT YOUR ACCESS CODE TO PROCEED WITH SENSITIVE OPERATIONS.
                </div>

                <form onSubmit={submit}>
                    <div className="mt-4">
                        <InputLabel htmlFor="password" value="ACCESS_CODE" className="text-slate-400 text-xs tracking-widest mb-1" />

                        <TextInput
                            id="password"
                            type="password"
                            name="password"
                            value={data.password}
                            className="mt-1 block w-full bg-slate-950 border-slate-700 text-slate-200 focus:border-cyan-500 focus:ring-cyan-500/50 placeholder-slate-600 rounded-md"
                            isFocused={true}
                            onChange={(e) => setData('password', e.target.value)}
                            placeholder="••••••••"
                        />

                        <InputError message={errors.password} className="mt-2 text-rose-400" />
                    </div>

                    <div className="mt-6 flex items-center justify-end">
                        <PrimaryButton 
                            className="ms-4 bg-cyan-600 hover:bg-cyan-500 text-white font-bold tracking-wider px-6 py-2 rounded-lg shadow-[0_0_10px_rgba(6,182,212,0.3)] transition-all border border-cyan-400" 
                            disabled={processing}
                        >
                            {processing ? 'VERIFYING...' : 'CONFIRM_ACCESS'}
                        </PrimaryButton>
                    </div>
                </form>
            </div>

            {/* Footer */}
            <div className="mt-8 text-[10px] text-slate-600 tracking-widest">
                SECURE SYSTEM V1.0 • AUTHORIZATION PROTOCOL
            </div>
        </div>
    );
}