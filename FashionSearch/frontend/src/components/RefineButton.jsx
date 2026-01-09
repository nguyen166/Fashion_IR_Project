import React from 'react';
import { Sparkles, ThumbsUp, ThumbsDown } from 'lucide-react';

const RefineButton = ({
    positiveCount,
    negativeCount,
    onRefine,
    isLoading,
    isVisible
}) => {
    if (!isVisible) return null;

    return (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50">
            <div className="bg-white rounded-2xl shadow-2xl border border-gray-200 p-4 flex items-center gap-4">
                {/* Selection Summary */}
                <div className="flex items-center gap-3 text-sm">
                    {positiveCount > 0 && (
                        <div className="flex items-center gap-1.5 text-green-600">
                            <ThumbsUp className="w-4 h-4" />
                            <span className="font-medium">{positiveCount}</span>
                        </div>
                    )}
                    {negativeCount > 0 && (
                        <div className="flex items-center gap-1.5 text-red-600">
                            <ThumbsDown className="w-4 h-4" />
                            <span className="font-medium">{negativeCount}</span>
                        </div>
                    )}
                </div>

                {/* Divider */}
                <div className="w-px h-8 bg-gray-200" />

                {/* Refine Button */}
                <button
                    onClick={onRefine}
                    disabled={isLoading}
                    className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-sky-600 
                     hover:from-blue-700 hover:to-sky-700 text-white font-semibold rounded-xl 
                     transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed
                     transform hover:scale-105 active:scale-95"
                >
                    {isLoading ? (
                        <>
                            <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                            <span>Refining...</span>
                        </>
                    ) : (
                        <>
                            <Sparkles className="w-5 h-5" />
                            <span>Refine Search</span>
                        </>
                    )}
                </button>

                {/* Clear Selection Button */}
                <button
                    onClick={() => window.dispatchEvent(new CustomEvent('clearSelection'))}
                    className="text-gray-400 hover:text-gray-600 text-sm underline"
                >
                    Clear
                </button>
            </div>
        </div>
    );
};

export default RefineButton;
