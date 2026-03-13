// Copyright (c) Microsoft Corporation.
//  Licensed under the MIT license.

import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { RefreshCw, Brain, TrendingUp, Activity } from 'lucide-react';

// 模拟从 JSON 加载数据的函数
const loadDataFromJSON = async (path) => {
  try {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Failed to load ${path}`);
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error loading data:', error);
    return [];
  }
};

// 生成医疗相关的时间序列数据（保留原逻辑）
const generateMedicalData = (type = 'ecg', length = 50, seed = 0) => {
  const data = [];
  let value = Math.random() * 10 + seed;

  for (let i = 0; i < length; i++) {
    let newValue = value;

    switch (type) {
      case 'ecg':
        if (i % 12 === 0) newValue += 15 + seed; // P波
        if (i % 12 === 3) newValue += 25 + seed * 0.5; // QRS复合波
        if (i % 12 === 7) newValue += 8 + seed * 0.3; // T波
        newValue += (Math.random() - 0.5) * 2;
        break;
      case 'eeg':
        newValue += Math.sin(i * 0.3 + seed) * 3 + Math.sin(i * 0.8 + seed) * 2 + (Math.random() - 0.5) * 1.5;
        break;
      case 'blood_pressure':
        newValue += Math.sin(i * 0.5 + seed) * 4 + (Math.random() - 0.5) * 3;
        newValue = Math.max(80, Math.min(160, newValue + 120 + seed * 5));
        break;
      case 'glucose':
        newValue += Math.sin(i * 0.1 + seed) * 2 + (Math.random() - 0.5) * 1;
        newValue = Math.max(70, Math.min(200, newValue + 100 + seed * 3));
        break;
      default:
        newValue += (Math.random() - 0.5) * 2;
    }

    value = newValue;
    data.push({
      time: i,
      value: Math.max(0, value)
    });
  }
  return data;
};

// 模拟不同模型在医疗数据上的训练结果
const simulateMedicalTraining = (data, modelType, isEnhanced = false, seed = 0) => {
  const baseAccuracies = {
    'rnn': 0.78,
    'cnn': 0.82,
    'transformer': 0.85
  };

  const baseAccuracy = baseAccuracies[modelType] || 0.75;
  const dataQualityBonus = isEnhanced ? 0.03 + Math.random() * 0.08 : 0;
  const randomVariation = (Math.random() - 0.5 + seed * 0.01) * 0.02;

  return Math.min(0.98, Math.max(0.65, baseAccuracy + dataQualityBonus + randomVariation));
};

const ModelThree = () => {
  const [originalData, setOriginalData] = useState([]);
  const [generatedData, setGeneratedData] = useState([]);
  const [selectedModel, setSelectedModel] = useState('rnn');
  const [isGenerating, setIsGenerating] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [accuracyResults, setAccuracyResults] = useState({
    original: 0.905,
    generated: 0.898,
    improvement: 0.007
  });


   // 添加针对不同疾病的精度结果
  const [accuracyResultsByDisease, setAccuracyResultsByDisease] = useState({
    myocardial_infarction: {
      // original: 0.82,
      // ours: 0.90532,
      // timegan: 0.78173,
      // timevae: 0.89445,
      // timevqvae: 0.89843,
      // diffusionts: 0.75979

      ours: 0.64429,
      timegan: 0.54113,
      timevae: 0.60053,
      timevqvae: 0.50578,
      diffusionts: 0.50039

    },
    alzheimers_disease: {
      // original: 0.78,
      ours: 0.77097,
      timegan: 0.51033,
      timevae: 0.68569,
      timevqvae: 0.55500,
      diffusionts: 0.47062
    },
    parkinsons_disease: {
      // original: 0.75,
      ours: 0.64444,
      timegan: 0.62780,
      timevae: 0.58565,
      timevqvae:  0.54659,
      diffusionts:  0.49319
    }
  });

  const [selectedDisease, setSelectedDisease] = useState('myocardial_infarction'); // 默认选择心肌梗死

  const medicalModels = [
          {
      id: 'rnn',
      name: 'Parkinson’s disease diagnosis',
      description: 'Use EEG (Electroencephalogram) signals to analyze cortical electrical activity, assess brain functional status, and identify the presence of Parkinson’s diseases.',
      icon: '⚡'
    },

    {
      id: 'cnn',
      name: 'Alzheimer’s disease diagnosis',
      description: 'Use EEG (Electroencephalogram) signals to analyze cortical electrical activity, assess brain functional status, and identify the presence of Alzheimer’s diseases.',
      icon: '🔍'
    },

          {
      id: 'transformer',
      name: 'Frontotemporal dementia',
      description: 'Use EEG (Electroencephalogram) signals to analyze cortical electrical activity, assess brain functional status, and identify the presence of Frontotemporal dementia.',
      icon: '🔄'
    },

  ];

  // 数据路径映射
  const dataPaths = {
        rnn: {
      original: '/json_data/ori_tdb_ori_TDB.json',
      generated: '/json_data/syn_tdb_syn_tdb.json'
    },

    cnn: {
      original: '/json_data/ori_apava_ori_apava.json',
      generated: '/json_data/syn_apava_syn_apava.json'
    },
    transformer: {
      original: '/json_data/ad_ori1_ad_ori1.json',
      generated: '/json_data/ad_syn1_ad_syn1.json'
    },
  };

  // 加载特定数据
  const loadSpecificData = async (modelId) => {
    const paths = dataPaths[modelId];
    const oriPath = paths.original;
    const genPath = paths.generated;

    // 尝试加载原始数据
    const oriData = await loadDataFromJSON(oriPath);
    const genData = await loadDataFromJSON(genPath);

    // 如果加载失败，用默认生成数据替代
    if (oriData.length === 0) {
      console.warn(`No data loaded from ${oriPath}, using synthetic data`);
      const synthetic = generateMedicalData(modelId === 'rnn' ? 'ecg' : 'eeg', 50);
      setOriginalData(synthetic);
    } else {
      setOriginalData(oriData);
    }

    if (genData.length === 0) {
      console.warn(`No data loaded from ${genPath}, using synthetic data`);
      const synthetic = generateMedicalData(modelId === 'rnn' ? 'ecg' : 'eeg', 50, 1);
      setGeneratedData(synthetic);
    } else {
      setGeneratedData(genData);
    }
  };

  // 切换模型时加载对应数据
  useEffect(() => {
    loadSpecificData(selectedModel);
  }, [selectedModel]);

  const handleGenerate = async () => {
    setIsGenerating(true);
    await new Promise(resolve => setTimeout(resolve, 2000));

    // 优化策略（保持原有逻辑）

    const paths = dataPaths[selectedModel];
    const genPath = paths.generated;

    // 尝试加载原始数据
    const genData = await loadDataFromJSON(genPath);

    const generateOptimizedData = (seed) => {
      return originalData.map((point, index) => {
        let optimizedValue = point.value;
        if (selectedModel === 'rnn') {
          optimizedValue += Math.sin(index * 0.2 + seed) * 1.2;
        } else if (selectedModel === 'cnn') {
          if (index % 10 < 3) optimizedValue *= (1.1 + seed * 0.02);
        } else if (selectedModel === 'transformer') {
          optimizedValue += Math.cos(index * 0.05 + seed) * 0.8;
        }
        if (Math.random() < 0.15) {
          optimizedValue += (Math.random() - 0.5 + seed * 0.1) * 3;
        }
        return {
          time: point.time,
          value: Math.max(0, optimizedValue + (Math.random() - 0.5) * 0.5)
        };
      });
    };

    // const generateOptimizedData = (seed) => {
    //   return originalData.map((point, index) => {
    //     let optimizedValue = point.value;
    //     if (selectedModel === 'rnn') {
    //       optimizedValue += Math.sin(index * 0.2 + seed) * 1.2;
    //     } else if (selectedModel === 'cnn') {
    //       if (index % 10 < 3) optimizedValue *= (1.1 + seed * 0.02);
    //     } else if (selectedModel === 'transformer') {
    //       optimizedValue += Math.cos(index * 0.05 + seed) * 0.8;
    //     }
    //     if (Math.random() < 0.15) {
    //       optimizedValue += (Math.random() - 0.5 + seed * 0.1) * 3;
    //     }
    //     return {
    //       time: point.time,
    //       value: Math.max(0, optimizedValue + (Math.random() - 0.5) * 0.5)
    //     };
    //   });
    // };

    if (genData.length === 0) {
      setGeneratedData(generateOptimizedData(1));
    } else {
      setGeneratedData(genData);
    }

    // setGeneratedData(genData);
    setIsGenerating(false);
  };

  const handleTraining = async () => {
    setIsTraining(true);
    await new Promise(resolve => setTimeout(resolve, 3000));

    const originalAccuracy = simulateMedicalTraining(originalData, selectedModel, false, 0);
    const generatedAccuracy = simulateMedicalTraining(generatedData, selectedModel, true, 1);

    setAccuracyResults({
      original: originalAccuracy,
      generated: generatedAccuracy,
      improvement: generatedAccuracy - originalAccuracy
    });
    setIsTraining(false);
  };

  const currentModel = medicalModels.find(m => m.id === selectedModel);

  // 合并数据用于对比图
  const combinedData = originalData.map((point, index) => ({
    time: point.time,
    original: point.value,
    generated: generatedData[index]?.value || 0
  }));

  // 准备精度对比数据

  // const accuracyComparisonData = [
  //   {
  //     dataset: 'Ours',
  //     accuracy: (accuracyResults.original * 100).toFixed(1),
  //     value: accuracyResults.original
  //   },
  //   {
  //     dataset: 'TimeGAN',
  //     accuracy: (accuracyResults.generated * 100).toFixed(1),
  //     value: accuracyResults.generated
  //   },
  //         {
  //     dataset: 'TimeVAE',
  //     accuracy: (accuracyResults.generated * 100).toFixed(1),
  //     value: accuracyResults.generated
  //   },
  //         {
  //     dataset: 'TimeVQVAE',
  //     accuracy: (accuracyResults.generated * 100).toFixed(1),
  //     value: accuracyResults.generated
  //   },
  //               {
  //     dataset: 'DiffusionTS',
  //     accuracy: (accuracyResults.generated * 100).toFixed(1),
  //     value: accuracyResults.generated
  //   },
  // ];


  // 根据当前疾病构建 accuracyComparisonData
  const currentResults = accuracyResultsByDisease[selectedDisease];
  const accuracyComparisonData = [
    { dataset: 'Ours', value: currentResults.ours },
    { dataset: 'TimeGAN', value: currentResults.timegan },
    { dataset: 'TimeVAE', value: currentResults.timevae },
    { dataset: 'TimeVQVAE', value: currentResults.timevqvae },
    { dataset: 'DiffusionTS', value: currentResults.diffusionts }
  ];


  // 找出精度最高的基线方法
  const baselineKeys = ['timegan', 'timevae', 'timevqvae', 'diffusionts']; // 👈 缺少这一行！

  let bestBaselineName = '';
  let bestBaselineAccuracy = -1; // 👈 改为 -1 更安全，避免所有精度都低于 0 时出错

  for (const key of baselineKeys) {
    const acc = currentResults[key];
    if (acc > bestBaselineAccuracy) {
      bestBaselineAccuracy = acc;
      // 使用映射确保名称正确（如 TimeGAN 而不是 Timegan）
      bestBaselineName = key
        .replace('timegan', 'TimeGAN')
        .replace('timevae', 'TimeVAE')
        .replace('timevqvae', 'TimeVQVAE')
        .replace('diffusionts', 'DiffusionTS');
    }
  }

  // 计算我们的方法比最强基线方法提高了多少
  const ourAccuracy = currentResults.ours;
  const improvementOverBestBaseline = ourAccuracy - bestBaselineAccuracy;

  // 现在你可以使用这两个变量：
  // bestBaselineName: 比如 "DiffusionTS"
  // improvementOverBestBaseline: 比如 0.04 (即 4%)

  // 在切换模型时同时考虑疾病的选择
  const handleModelChange = (modelId) => {
    setSelectedModel(modelId);
    // 确保疾病选择与模型匹配，例如，这里只是简单地处理了默认情况
    if (modelId === 'rnn') setSelectedDisease('myocardial_infarction');
    else if (modelId === 'cnn') setSelectedDisease('alzheimers_disease');
    else if (modelId === 'transformer') setSelectedDisease('parkinsons_disease');
  };

  // 更新下游模型选择部分的点击事件处理函数
  // const currentModel = medicalModels.find(m => m.id === selectedModel);

  // const accuracyComparisonData = [
  //   {
  //     dataset: 'Original',
  //     accuracy: (accuracyResults.original * 100).toFixed(1),
  //     value: accuracyResults.original
  //   },
  //   {
  //     dataset: 'Generated',
  //     accuracy: (accuracyResults.generated * 100).toFixed(1),
  //     value: accuracyResults.generated
  //   }
  // ];


  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-900">Time-series Generation for Medical Diagnosis Tasks</h2>
          <button
            onClick={() => {
              setOriginalData([]);
              setGeneratedData([]);
              loadSpecificData(selectedModel);
            }}
            className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors flex items-center space-x-2"

          >
            <RefreshCw size={16} />
            <span>One-click run</span>
          </button>
        </div>

        {/* 下游模型选择 */}
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-4 mb-6">
          <div className="flex items-center space-x-2 mb-3">
            <Brain size={20} className="text-purple-600" />
            <h3 className="text-lg font-semibold text-gray-800">Medical diagnosis downstream task selection</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
            {medicalModels.map((model) => (
              <button
                key={model.id}
                onClick={() => handleModelChange(model.id)}
                className={`p-4 rounded-lg border-2 transition-colors text-left ${
                  selectedModel === model.id
                    ? 'border-purple-500 bg-purple-50 text-purple-800'
                    : 'border-gray-200 bg-white hover:border-gray-300'
                }`}
              >
                <div className="flex items-center space-x-2 mb-2">
                  <span className="text-2xl">{model.icon}</span>
                  <div className="font-semibold">{model.name}</div>
                </div>
                <div className="text-sm text-gray-600">{model.description}</div>
              </button>
            ))}
          </div>

          <div className="bg-white rounded-lg p-4 mb-4">
            <div className="flex items-center space-x-2 mb-2">
              <Activity size={18} className="text-red-500" />
              <h4 className="font-semibold text-gray-800">Currently selected：{currentModel?.name}</h4>
            </div>
            <p className="text-sm text-gray-600">{currentModel?.description}</p>
            <p className="text-sm text-blue-600 mt-1">Intelligent diagnosis of medical diseases through the analysis of ECG or EEG signals.</p>
          </div>

          <div className="flex space-x-3">
            <button
              onClick={handleGenerate}
              disabled={isGenerating}
              className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-purple-300 transition-colors flex items-center space-x-2"
            style={{ pointerEvents: 'none' }}
            >

              {isGenerating ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                  <span className="flex-1 min-w-0 text-left leading-tight break-words"> Generating optimized diagnostic sequences for {currentModel?.name}...</span>
                </>
              ) : (
                <>
                  <RefreshCw size={20} />
                  <span className="flex-1 min-w-0 text-left leading-tight break-words">Generate 100 diagnosis sequences guided by the well-trained basic model’s gradients and randomly display one example</span>
                </>
              )}
            </button>

            <button
              onClick={handleTraining}
              disabled={isTraining}
              className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-green-300 transition-colors flex items-center space-x-2"
            style={{ pointerEvents: 'none' }}
            >
              {isTraining ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                  <span className="flex-1 min-w-0 text-left leading-tight break-words">Retrain a diagnostic model through augmented training set...</span>
                </>
              ) : (
                <>
                  <TrendingUp size={20} />
                  <span className="flex-1 min-w-0 text-left leading-tight break-words">Augment the original training set with these new sequences, retrain the disease-diagnosis model, and compare its accuracy with baseline models.</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* 原始序列和生成序列展示 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-gray-800 mb-3">Original medical time series</h3>
            <div className="text-sm text-gray-600 mb-2">{currentModel?.name}</div>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={originalData}>
                  <CartesianGrid strokeDasharray="3 3"/>
                  <XAxis dataKey="time"/>
                  <YAxis/>
                  <Tooltip/>
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#3B82F6"
                    strokeWidth={2}
                    dot={false}
                    name="Original medical time series"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-gray-800 mb-3">Gradient-guided time series</h3>
            <div className="text-sm text-gray-600 mb-2">Generated time series for {currentModel?.name}</div>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={generatedData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#10B981"
                    strokeWidth={2}
                    dot={false}
                    name="Gradient-guided time series"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* 对比图 */}
        <div className="bg-gray-50 rounded-lg p-4 mb-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-3">Overall comparison between the original and generated time series</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={combinedData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="original"
                  stroke="#3B82F6"
                  strokeWidth={3}
                  name="Original medical time series"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="generated"
                  stroke="#10B981"
                  strokeWidth={2}
                  name="Gradient-guided time series"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 疾病诊断精度对比 */}
        <div className="bg-gradient-to-r from-red-50 to-orange-50 rounded-lg p-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center space-x-2">
            <Activity size={24} className="text-red-600" />
            <span>Disease diagnosis model accuracy comparison</span>
          </h3>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* 条形图 */}
            <div className="bg-white rounded-lg p-4">
              <h4 className="text-lg font-semibold text-gray-800 mb-3">{currentModel?.name}</h4>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={accuracyComparisonData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="dataset" />
                    <YAxis domain={['auto', 'auto']} />
                    <Tooltip formatter={(value) => [`${(value * 100).toFixed(1)}%`, 'AUROC']} />
                    <Bar dataKey="value" fill="#ef4444" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* 详细指标 */}
            <div className="space-y-3">
              <div className="bg-white rounded-lg p-4">
                <h4 className="font-semibold text-blue-800 mb-1">Diagnostic accuracy with <span
                    className="bg-yellow-200 text-green-900 px-1 rounded font-bold">
    ours
  </span> synthetic data</h4>
                <div className="text-xl font-bold text-blue-600">
                  {(ourAccuracy * 100).toFixed(1)}%
                </div>
                <p className="text-xs text-gray-600">Our approach takes into account the gradients of different disease models, thereby specifically generating datasets that help improve model accuracy.</p>
              </div>

              <div className="bg-white rounded-lg p-3">
                <h4 className="font-semibold text-green-800 mb-1 text-sm">Diagnostic accuracy with best competitor  <span className="bg-yellow-200 text-green-900 px-1 rounded font-bold">
    {bestBaselineName}
  </span>

                  's synthetic data</h4>
                <div className="text-lg font-bold text-green-600">
                  {(bestBaselineAccuracy* 100).toFixed(1)}%
                  <span className="text-sm text-green-500 ml-1">
                    (+{(improvementOverBestBaseline * 100).toFixed(1)}%)
                  </span>
                </div>
                <p className="text-xs text-gray-600">The baseline model concentrates on capturing the original data distribution but neglects the gradient signals from downstream tasks, thus failing to generate data that can effectively enhance model accuracy.</p>
              </div>
            </div>
          </div>

          {/* 训练详情 */}
          <div className="mt-6 grid grid-cols-1 md:grid-cols-4 gap-4">
            {/*<div className="bg-white rounded-lg p-4">*/}
            {/*  <h5 className="font-semibold text-gray-800">Diagnostic Model</h5>*/}
            {/*  <p className="text-sm text-gray-600 mt-1">*/}
            {/*    {currentModel?.name}*/}
            {/*  </p>*/}
            {/*  <p className="text-xs text-gray-500">*/}
            {/*    {currentModel?.icon} {currentModel?.description.slice(0, 15)}...*/}
            {/*  </p>*/}
            {/*</div>*/}
            <div className="bg-white rounded-lg p-4">
              <h5 className="font-semibold text-gray-800">Medical Data Type</h5>
              <p className="text-sm text-gray-600 mt-1">
                {selectedModel === 'rnn' ? 'ECG' : 'EEG'} data
              </p>
              <p className="text-xs text-gray-500">
                {selectedModel === 'rnn' ? 'Cardiovascular disease' : 'Neurodegenerative disease'} diagnosis
              </p>
            </div>
            <div className="bg-white rounded-lg p-4">
              <h5 className="font-semibold text-gray-800">Best Performance</h5>
              <p className="text-sm text-gray-600 mt-1">
                Trained with our synthetic data
              </p>
              <p className="text-xs text-gray-500">
                Accuracy: {(ourAccuracy * 100).toFixed(1)}%
              </p>
            </div>
            <div className="bg-white rounded-lg p-4">
              <h5 className="font-semibold text-gray-800">Training Status</h5>
              <p className="text-sm text-gray-600 mt-1">
                {isTraining ? 'Training in Progress...' : 'Training Completed'}
              </p>
              <p className="text-xs text-gray-500">
                {/*Training set: {originalData.length + generatedData.length} samples*/}
                {/*Training set: 100 samples*/}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelThree;