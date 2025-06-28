import React from "react";

export default function Home() {
  return (
    <div className="max-w-5xl mx-auto">
      <div className="text-2xl font-bold mb-8 text-gray-900">Consumer Review Insights & Actions</div>
      <div className="bg-white rounded-xl shadow p-8">
        <div className="text-xl font-semibold mb-4">Main Form Placeholder</div>
        <p className="text-gray-600 mb-2">This is where the main form and features will go.</p>
        {/* TODO: Add product selection, review extraction, LLM selection, etc. */}
      </div>
    </div>
  );
}
