import React from 'react';
import ImageCard from './ImageCard';
import { Search, ImageOff } from 'lucide-react';

const ImageGrid = ({
    results,
    positiveIds,
    negativeIds,
    onLike,
    onDislike,
    isLoading,
    hasSearched
}) => {
    // Loading state
    if (isLoading) {
        return (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                {[...Array(10)].map((_, i) => (
                    <div key={i} className="animate-pulse">
                        <div className="aspect-square bg-gray-200 rounded-xl mb-2" />
                        <div className="h-4 bg-gray-200 rounded w-3/4 mb-1" />
                        <div className="h-3 bg-gray-200 rounded w-1/2" />
                    </div>
                ))}
            </div>
        );
    }

    // Empty state - before search
    if (!hasSearched) {
        return (
            <div className="flex flex-col items-center justify-center py-20 text-gray-400">
                <Search className="w-16 h-16 mb-4" />
                <h3 className="text-xl font-medium text-gray-600">Start Your Fashion Search</h3>
                <p className="text-gray-500 mt-2 text-center max-w-md">
                    Enter a description like "red dress" or "blue sneakers",
                    or upload an image to find similar items.
                </p>
            </div>
        );
    }

    // Empty results state
    if (results.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center py-20 text-gray-400">
                <ImageOff className="w-16 h-16 mb-4" />
                <h3 className="text-xl font-medium text-gray-600">No Results Found</h3>
                <p className="text-gray-500 mt-2">Try a different search term or image.</p>
            </div>
        );
    }

    // Results grid
    return (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {results.map((item) => (
                <ImageCard
                    key={item.id}
                    item={item}
                    isPositive={positiveIds.includes(item.id)}
                    isNegative={negativeIds.includes(item.id)}
                    onLike={onLike}
                    onDislike={onDislike}
                />
            ))}
        </div>
    );
};

export default ImageGrid;
