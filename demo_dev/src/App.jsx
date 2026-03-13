// Copyright (c) Microsoft Corporation.
//  Licensed under the MIT license.

import React, { useState } from 'react';
import { Play, FileText, Activity } from 'lucide-react';
import ModelOne from './components/ModelOne';
import ModelTwo from './components/ModelTwo.jsx';
import ModelThree from './components/ModelThree';

function App() {
  const [activeModel, setActiveModel] = useState('model1');

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <h1 className="text-3xl font-bold text-gray-900">Time Series Generation Model Demonstration System</h1>
            <div className="flex space-x-4">
              <button
                onClick={() => setActiveModel('model1')}
                className={`px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors ${
                  activeModel === 'model1' 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                <Play size={18} />
                <span>Domain prompt guided generation</span>
              </button>
              <button
                onClick={() => setActiveModel('model2')}
                className={`px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors ${
                  activeModel === 'model2' 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                <FileText size={18} />
                <span>Text-controlled generation</span>
              </button>
              <button
                onClick={() => setActiveModel('model3')}
                className={`px-4 py-2 rounded-lg flex items-center space-x-2 transition-colors ${
                  activeModel === 'model3' 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                <Activity size={18} />
                <span>Gradient-guided generation</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeModel === 'model1' && <ModelOne />}
        {activeModel === 'model2' && <ModelTwo />}
        {activeModel === 'model3' && <ModelThree />}
      </main>
    </div>
  );
}

export default App;