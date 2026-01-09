import React from 'react';
import { ThumbsUp, ThumbsDown } from 'lucide-react';
import { getImageUrl } from '../api/searchApi';

const ImageCard = ({
    item,
    isPositive,
    isNegative,
    onLike,
    onDislike
}) => {
    const imageUrl = getImageUrl(item.image_path);

    // Determine border color based on selection state
    let borderClass = 'border-transparent';
    if (isPositive) {
        borderClass = 'border-green-500 ring-2 ring-green-300';
    } else if (isNegative) {
        borderClass = 'border-red-500 ring-2 ring-red-300';
    }

    return (
        <div
            className={`group relative bg-white rounded-xl overflow-hidden shadow-md 
                  border-4 ${borderClass} transition-all duration-200 hover:shadow-xl`}
        >
            {/* Image */}
            <div className="aspect-square overflow-hidden bg-gray-100">
                <img
                    src={imageUrl}
                    alt={item.productDisplayName || 'Fashion item'}
                    className="w-full h-full object-cover transition-transform duration-300 
                     group-hover:scale-105"
                    onError={(e) => {
                        e.target.src = 'https://via.placeholder.com/300x300?text=No+Image';
                    }}
                />
            </div>

            {/* Hover Overlay with Buttons */}
            <div className="absolute inset-0 bg-black/0 group-hover:bg-black/40 
                      transition-colors duration-200 flex items-center justify-center">
                <div className="flex gap-3 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                    {/* Like Button */}
                    <button
                        onClick={() => onLike(item.id)}
                        className={`p-3 rounded-full transition-all duration-200 transform hover:scale-110
                       ${isPositive
                                ? 'bg-green-500 text-white'
                                : 'bg-white/90 text-green-600 hover:bg-green-500 hover:text-white'
                            }`}
                        title="Like - More like this"
                    >
                        <ThumbsUp className="w-6 h-6" />
                    </button>

                    {/* Dislike Button */}
                    <button
                        onClick={() => onDislike(item.id)}
                        className={`p-3 rounded-full transition-all duration-200 transform hover:scale-110
                       ${isNegative
                                ? 'bg-red-500 text-white'
                                : 'bg-white/90 text-red-600 hover:bg-red-500 hover:text-white'
                            }`}
                        title="Dislike - Less like this"
                    >
                        <ThumbsDown className="w-6 h-6" />
                    </button>
                </div>
            </div>

            {/* Selection Indicator Badge */}
            {(isPositive || isNegative) && (
                <div className={`absolute top-2 right-2 p-1.5 rounded-full 
                        ${isPositive ? 'bg-green-500' : 'bg-red-500'}`}>
                    {isPositive ? (
                        <ThumbsUp className="w-4 h-4 text-white" />
                    ) : (
                        <ThumbsDown className="w-4 h-4 text-white" />
                    )}
                </div>
            )}

            {/* Item Info */}
            <div className="p-3">
                <h3 className="text-sm font-medium text-gray-800 truncate">
                    {item.productDisplayName || 'Unknown Product'}
                </h3>
                <div className="flex flex-wrap gap-1 mt-2">
                    {item.category && (
                        <span className="px-2 py-0.5 bg-blue-100 text-blue-700 text-xs rounded-full">
                            {item.category}
                        </span>
                    )}
                    {item.gender && item.gender !== 'Unknown' && (
                        <span className="px-2 py-0.5 bg-purple-100 text-purple-700 text-xs rounded-full">
                            {item.gender}
                        </span>
                    )}
                    {item.baseColour && item.baseColour !== 'Unknown' && (
                        <span className="px-2 py-0.5 bg-orange-100 text-orange-700 text-xs rounded-full">
                            {item.baseColour}
                        </span>
                    )}
                    {item.season && item.season !== 'Unknown' && (
                        <span className="px-2 py-0.5 bg-green-100 text-green-700 text-xs rounded-full">
                            {item.season}
                        </span>
                    )}
                </div>
            </div>
        </div>
    );
};

export default ImageCard;
