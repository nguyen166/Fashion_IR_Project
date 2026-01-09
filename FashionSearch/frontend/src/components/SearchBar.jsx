import React, { useRef } from 'react';
import { Search, Upload, X, Image as ImageIcon, ChevronDown } from 'lucide-react';

const SearchBar = ({
    query,
    setQuery,
    onSearch,
    onImageUpload,
    uploadedImage,
    setUploadedImage,
    isLoading,
    categories,
    selectedCategory,
    setSelectedCategory
}) => {
    const fileInputRef = useRef(null);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (query.trim() || uploadedImage) {
            onSearch();
        }
    };

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (file && file.type.startsWith('image/')) {
            setUploadedImage(file);
            onImageUpload(file);
        }
    };

    const handleUploadClick = () => {
        fileInputRef.current?.click();
    };

    const clearImage = () => {
        setUploadedImage(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    return (
        <div className="w-full max-w-4xl mx-auto">
            <form onSubmit={handleSubmit} className="flex flex-col gap-4">
                {/* Search Input Row */}
                <div className="flex gap-3 flex-wrap">
                    {/* Text Input */}
                    <div className="flex-1 min-w-[200px] relative">
                        <input
                            type="text"
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            placeholder="Search for fashion items... (e.g., 'red dress', 'blue jeans')"
                            className="w-full px-5 py-4 pl-12 text-lg border-2 border-gray-200 rounded-xl 
                         focus:border-blue-500 focus:outline-none transition-colors
                         bg-white shadow-sm"
                            disabled={isLoading}
                        />
                        <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400 w-5 h-5" />
                    </div>

                    {/* Category Dropdown */}
                    <div className="relative">
                        <select
                            value={selectedCategory || ''}
                            onChange={(e) => setSelectedCategory(e.target.value || null)}
                            className="appearance-none px-5 py-4 pr-10 border-2 border-gray-200 rounded-xl 
                                     bg-white shadow-sm text-gray-700 cursor-pointer
                                     focus:border-blue-500 focus:outline-none transition-colors
                                     disabled:opacity-50"
                            disabled={isLoading}
                        >
                            <option value="">All Categories</option>
                            {categories.map((cat) => (
                                <option key={cat} value={cat}>{cat}</option>
                            ))}
                        </select>
                        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 w-5 h-5 pointer-events-none" />
                    </div>

                    {/* Upload Button */}
                    <button
                        type="button"
                        onClick={handleUploadClick}
                        className="px-6 py-4 bg-gray-100 hover:bg-gray-200 border-2 border-gray-200 
                       rounded-xl transition-colors flex items-center gap-2 text-gray-700
                       disabled:opacity-50"
                        disabled={isLoading}
                    >
                        <Upload className="w-5 h-5" />
                        <span className="hidden sm:inline">Upload</span>
                    </button>
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept="image/*"
                        onChange={handleFileChange}
                        className="hidden"
                    />

                    {/* Search Button */}
                    <button
                        type="submit"
                        disabled={isLoading || (!query.trim() && !uploadedImage)}
                        className="px-8 py-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold 
                       rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed
                       flex items-center gap-2"
                    >
                        {isLoading ? (
                            <>
                                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                                <span className="hidden sm:inline">Searching...</span>
                            </>
                        ) : (
                            <>
                                <Search className="w-5 h-5" />
                                <span className="hidden sm:inline">Search</span>
                            </>
                        )}
                    </button>
                </div>

                {/* Uploaded Image Preview */}
                {uploadedImage && (
                    <div className="flex items-center gap-3 p-3 bg-blue-50 rounded-xl border border-blue-200">
                        <ImageIcon className="w-5 h-5 text-blue-600" />
                        <img
                            src={URL.createObjectURL(uploadedImage)}
                            alt="Uploaded"
                            className="w-16 h-16 object-cover rounded-lg"
                        />
                        <div className="flex-1">
                            <p className="text-sm font-medium text-blue-800">{uploadedImage.name}</p>
                            <p className="text-xs text-blue-600">
                                {(uploadedImage.size / 1024).toFixed(1)} KB
                            </p>
                        </div>
                        <button
                            type="button"
                            onClick={clearImage}
                            className="p-2 hover:bg-blue-100 rounded-full transition-colors"
                        >
                            <X className="w-5 h-5 text-blue-600" />
                        </button>
                    </div>
                )}
            </form>
        </div>
    );
};

export default SearchBar;
