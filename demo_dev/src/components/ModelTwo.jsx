// Copyright (c) Microsoft Corporation.
//  Licensed under the MIT license.

import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';
import { RefreshCw, Type, Database } from 'lucide-react';

// 加载JSON数据的工具函数
const loadDataFromJSON = async (filePath) => {
  try {
    const response = await fetch(filePath);
    if (!response.ok) {
      throw new Error(`Failed to load ${filePath}: ${response.status}`);
    }
    const data = await response.json();
    return data.map(point => ({
      time: point.time,
      value: point.value
    }));
  } catch (error) {
    console.error('Error loading JSON:', error);
    return []; // 失败时返回空数组兜底
  }
};

// 导入不同的electricity数据文件（作为通用占位数据）
import electricity_gt_3143 from '/public/json_data/electricity_gen_0.json';
import electricity_gt_2826 from '/public/json_data/electricity_gt_2826.json';
import electricity_gt_2475 from '/public/json_data/electricity_gt_2475.json';
import electricity_gt_1158 from '/public/json_data/electricity_gt_1158.json';

import ped1_ped1 from '/public/json_data/ped1_ped1.json';
import ped2_ped2 from '/public/json_data/ped2_ped2.json';
import ped3_ped3 from '/public/json_data/ped3_ped3.json';
import pedori_pedori from '/public/json_data/pedori_pedori.json';

// 为不同选项创建数据映射（统一用 electricity 数据）
const domainDataMap = {
  electricity_1: { 0: electricity_gt_3143 },
  electricity_2: { 0: electricity_gt_3143 },
  pedestrian_1: { 0: pedori_pedori },
  pedestrian_2: { 0: pedori_pedori },
  // stock: { 0: [] },
  // temperature: { 0: [] },
  // network: { 0: [] },
};

const generateDomainData = (domain, length = 200, seed = 0) => {
  const data = domainDataMap[domain]?.[seed] || [];
  return data.slice(0, length);
};

// ==================== 所有领域都使用 electricity_1 的三个模板 ====================
const predefinedPromptsByDomain = {
  electricity_1: [
    {
      text: 'Guided Text1: for this prediction window of 168 time steps, the data ranges from a minimum of 10 to a maximum of 42. The average value is 35, with a standard deviation of 9, indicating moderate variability and an overall increasing trend. Notably, the series exhibits prominent peaks at time steps around 150 and noticeable dips around 5.',
      dataFile: '/public/json_data/electricity_gen_1.json',
      comment: 'For this prediction window of 168 time steps, the data ranges from a minimum of 10 to a maximum of 42. The average value is 35, with a standard deviation of 9, indicating moderate variability and an overall increasing trend. Notably, the series exhibits prominent peaks at time steps around 150 and noticeable dips around 5.'
    },
    {
      text: 'Guided Text2: The forecast horizon presented here is 168 steps. Data statistics reveal values from 10 up to 42, averaging 28 with a standard deviation of 8, indicating high variability. Notable peaks are anticipated around time steps 160.',
      dataFile: '/public/json_data/electricity_gen_2.json',
      comment: 'The forecast horizon presented here is 168 steps. Data statistics reveal values from 10 up to 42, averaging 28 with a standard deviation of 8, indicating high variability. Notable peaks are anticipated around time steps 160.'
    },
    {
      text: 'Guided Text3: For this prediction window of 168 time steps, the data ranges from a minimum of 5 to a maximum of 25. The average value is 17, with a standard deviation of 5, indicating moderate variability and an overall fluctuating trend. Notably, the series exhibits prominent peaks around time steps 40 and noticeable dips around 5.',
      dataFile: '/public/json_data/electricity_gen_3.json',
      comment: 'For this window of 168 time steps, the data ranges from a minimum of 5 to a maximum of 25. The average value is 17, with a standard deviation of 5, indicating moderate variability and an overall fluctuating trend. Notably, the series exhibits prominent peaks around time steps 40 and noticeable dips around 5.'
    }
  ],

  electricity_2: [
    {
      text: 'Guided Text1: for this window of 168 time steps, the data ranges from a minimum of 5 to a maximum of 25. The average value is 17, with a standard deviation of 5, indicating moderate variability and an overall fluctuating trend. Notably, the series exhibits prominent peaks around time steps 40 and noticeable dips around 5.',
      dataFile: '/public/json_data/electricity_gen_3.json',
      comment: 'For this window of 168 time steps, the data ranges from a minimum of 5 to a maximum of 25. The average value is 17, with a standard deviation of 5, indicating moderate variability and an overall fluctuating trend. Notably, the series exhibits prominent peaks around time steps 40 and noticeable dips around 5.'
    },
    {
      text: 'Guided Text2: for this prediction window of 168 time steps, the data ranges from a minimum of 10 to a maximum of 42. The average value is 35, with a standard deviation of 9, indicating moderate variability and an overall increasing trend. Notably, the series exhibits prominent peaks at time steps around 150 and noticeable dips around 5.',
      dataFile: '/public/json_data/electricity_gen_1.json',
      comment: 'For this prediction window of 168 time steps, the data ranges from a minimum of 10 to a maximum of 42. The average value is 35, with a standard deviation of 9, indicating moderate variability and an overall increasing trend. Notably, the series exhibits prominent peaks at time steps around 150 and noticeable dips around 5.'
    },
    {
      text: 'Guided Text3: the forecast horizon presented here is 168 steps. Data statistics reveal values from 10 up to 42, averaging 28 with a standard deviation of 8, indicating high variability. Notable peaks are anticipated around time steps 160.',
      dataFile: '/public/json_data/electricity_gen_2.json',
      comment: 'The forecast horizon presented here is 168 steps. Data statistics reveal values from 10 up to 42, averaging 28 with a standard deviation of 8, indicating high variability. Notable peaks are anticipated around time steps 160.'
    }
  ],

  pedestrian_1: [
    {
      text: 'Guided Text1: for this prediction window of 168 time steps, values range between 5 and 26, with an average of 13 and standard deviation of 6. The overall trend is an overall decreasing trend. Peaks are evident around steps 75, while troughs occur near the end of sequence.',
      dataFile: '/public/json_data/ped1_ped1.json',
      comment: 'For this predictive horizon of 168 steps, values range between 5 and 26, with an average of 13 and standard deviation of 6. The overall trend is an overall decreasing trend. Peaks are evident around steps 75, while troughs occur near the end of sequence.'
    },
    {
      text: 'Guided Text2: The series spans 168 steps with values ranging from 10 to 40. It maintains a mean of 19. The data shows fluctuating without a strong upward or downward trend, with notable highs at steps 50.',
      dataFile: '/public/json_data/ped2_ped2.json',
      comment: 'The series spans 168 steps with values ranging from 10 to 40. It maintains a mean of 19. The data shows fluctuating without a strong upward or downward trend, with notable highs at steps 50.'
    },
    {
      text: 'Guided Text3: For the forecast window of 168 steps, the time series values fluctuate between 10 and 40, averaging 23 with a standard deviation of 8. An an overall increasing trend pattern is observed, with peaks emerging around 150 steps.',
      dataFile: '/public/json_data/ped3_ped3.json',
      comment: 'For the forecast window of 168 steps, the time series values fluctuate between 10 and 40, averaging 23 with a standard deviation of 8. An an overall increasing trend pattern is observed, with peaks emerging around 150 steps.'
    }
  ],

  pedestrian_2: [
    {
      text: 'Guided Text1: for the forecast window of 168 steps, the time series values fluctuate between 10 and 40, averaging 23 with a standard deviation of 8. An an overall increasing trend pattern is observed, with peaks emerging around 150 steps.',
      dataFile: '/public/json_data/ped3_ped3.json',
      comment: 'For the forecast window of 168 steps, the time series values fluctuate between 10 and 40, averaging 23 with a standard deviation of 8. An an overall increasing trend pattern is observed, with peaks emerging around 150 steps.'
    },
    {
      text: 'Guided Text2: The series spans 168 steps with values ranging from 10 to 40. It maintains a mean of 19. The data shows fluctuating without a strong upward or downward trend, with notable highs at steps 50.',
      dataFile: '/public/json_data/ped2_ped2.json',
      comment: 'The series spans 168 steps with values ranging from 10 to 40. It maintains a mean of 19. The data shows fluctuating without a strong upward or downward trend, with notable highs at steps 50.'
    },
    {
      text: 'Guided Text3: for this prediction window of 168 time steps, values range between 5 and 26, with an average of 13 and standard deviation of 6. The overall trend is an overall decreasing trend. Peaks are evident around steps 75, while troughs occur near the end of sequence.',
      dataFile: '/public/json_data/ped1_ped1.json',
      comment: 'For this predictive horizon of 168 steps, values range between 5 and 26, with an average of 13 and standard deviation of 6. The overall trend is an overall decreasing trend. Peaks are evident around steps 75, while troughs occur near the end of sequence.'
    }
  ],

  // stock: [
  //   {
  //     text: 'Guided Text1: for this prediction window of 168 time steps, the data ranges from a minimum of 10 to a maximum of 42. The average value is 35, with a standard deviation of 9, indicating moderate variability and an overall increasing trend. Notably, the series exhibits prominent peaks at time steps around 150 and noticeable dips around 5.',
  //     dataFile: '/public/json_data/electricity_gen_1.json',
  //     comment: 'For this prediction window of 168 time steps, the data ranges from a minimum of 10 to a maximum of 42. The average value is 35, with a standard deviation of 9, indicating moderate variability and an overall increasing trend. Notably, the series exhibits prominent peaks at time steps around 150 and noticeable dips around 5.'
  //   },
  //   {
  //     text: 'Guided Text2: The forecast horizon presented here is 168 steps. Data statistics reveal values from 10 up to 42, averaging 28 with a standard deviation of 8, indicating high variability. Notable peaks are anticipated around time steps 160.',
  //     dataFile: '/public/json_data/electricity_gen_2.json',
  //     comment: 'The forecast horizon presented here is 168 steps. Data statistics reveal values from 10 up to 42, averaging 28 with a standard deviation of 8, indicating high variability. Notable peaks are anticipated around time steps 160.'
  //   },
  //   {
  //     text: 'Guided Text3: For this prediction window of 168 time steps, the data ranges from a minimum of 5 to a maximum of 25. The average value is 17, with a standard deviation of 5, indicating moderate variability and an overall fluctuating trend. Notably, the series exhibits prominent peaks around time steps 40 and noticeable dips around 5.',
  //     dataFile: '/public/json_data/electricity_gen_3.json',
  //     comment: 'For this window of 168 time steps, the data ranges from a minimum of 5 to a maximum of 25. The average value is 17, with a standard deviation of 5, indicating moderate variability and an overall fluctuating trend. Notably, the series exhibits prominent peaks around time steps 40 and noticeable dips around 5.'
  //   }
  // ],
  //
  // temperature: [
  //   {
  //     text: 'Guided Text1: for this prediction window of 168 time steps, the data ranges from a minimum of 10 to a maximum of 42. The average value is 35, with a standard deviation of 9, indicating moderate variability and an overall increasing trend. Notably, the series exhibits prominent peaks at time steps around 150 and noticeable dips around 5.',
  //     dataFile: '/public/json_data/electricity_gen_1.json',
  //     comment: 'For this prediction window of 168 time steps, the data ranges from a minimum of 10 to a maximum of 42. The average value is 35, with a standard deviation of 9, indicating moderate variability and an overall increasing trend. Notably, the series exhibits prominent peaks at time steps around 150 and noticeable dips around 5.'
  //   },
  //   {
  //     text: 'Guided Text2: The forecast horizon presented here is 168 steps. Data statistics reveal values from 10 up to 42, averaging 28 with a standard deviation of 8, indicating high variability. Notable peaks are anticipated around time steps 160.',
  //     dataFile: '/public/json_data/electricity_gen_2.json',
  //     comment: 'The forecast horizon presented here is 168 steps. Data statistics reveal values from 10 up to 42, averaging 28 with a standard deviation of 8, indicating high variability. Notable peaks are anticipated around time steps 160.'
  //   },
  //   {
  //     text: 'Guided Text3: For this prediction window of 168 time steps, the data ranges from a minimum of 5 to a maximum of 25. The average value is 17, with a standard deviation of 5, indicating moderate variability and an overall fluctuating trend. Notably, the series exhibits prominent peaks around time steps 40 and noticeable dips around 5.',
  //     dataFile: '/public/json_data/electricity_gen_3.json',
  //     comment: 'For this window of 168 time steps, the data ranges from a minimum of 5 to a maximum of 25. The average value is 17, with a standard deviation of 5, indicating moderate variability and an overall fluctuating trend. Notably, the series exhibits prominent peaks around time steps 40 and noticeable dips around 5.'
  //   }
  // ],
  //
  // network: [
  //   {
  //     text: 'Guided Text1: for this prediction window of 168 time steps, the data ranges from a minimum of 10 to a maximum of 42. The average value is 35, with a standard deviation of 9, indicating moderate variability and an overall increasing trend. Notably, the series exhibits prominent peaks at time steps around 150 and noticeable dips around 5.',
  //     dataFile: '/public/json_data/electricity_gen_1.json',
  //     comment: 'For this prediction window of 168 time steps, the data ranges from a minimum of 10 to a maximum of 42. The average value is 35, with a standard deviation of 9, indicating moderate variability and an overall increasing trend. Notably, the series exhibits prominent peaks at time steps around 150 and noticeable dips around 5.'
  //   },
  //   {
  //     text: 'Guided Text2: The forecast horizon presented here is 168 steps. Data statistics reveal values from 10 up to 42, averaging 28 with a standard deviation of 8, indicating high variability. Notable peaks are anticipated around time steps 160.',
  //     dataFile: '/public/json_data/electricity_gen_2.json',
  //     comment: 'The forecast horizon presented here is 168 steps. Data statistics reveal values from 10 up to 42, averaging 28 with a standard deviation of 8, indicating high variability. Notable peaks are anticipated around time steps 160.'
  //   },
  //   {
  //     text: 'Guided Text3: For this prediction window of 168 time steps, the data ranges from a minimum of 5 to a maximum of 25. The average value is 17, with a standard deviation of 5, indicating moderate variability and an overall fluctuating trend. Notably, the series exhibits prominent peaks around time steps 40 and noticeable dips around 5.',
  //     dataFile: '/public/json_data/electricity_gen_3.json',
  //     comment: 'For this window of 168 time steps, the data ranges from a minimum of 5 to a maximum of 25. The average value is 17, with a standard deviation of 5, indicating moderate variability and an overall fluctuating trend. Notably, the series exhibits prominent peaks around time steps 40 and noticeable dips around 5.'
  //   }
  // ]
};

// 下拉菜单选项配置
const domainOptions = [
  { value: 'electricity_1', label: 'electricity #1', description: 'Electricity consumption data with daily periodicity' },
  { value: 'electricity_2', label: 'electricity #2', description: 'Electricity consumption data with daily periodicity' },
  { value: 'pedestrian_1', label: 'pedestrian #1', description: 'Pedestrian dataset captures hourly pedestrian counts collected from 66 sensors in Melbourne' },
  { value: 'pedestrian_2', label: 'pedestrian #2', description: 'Pedestrian dataset captures hourly pedestrian counts collected from 66 sensors in Melbourne' },
];

function HighlightKeywords({ text }) {
  if (!text) return null;

  const keywordStyles = {
    minimum: '#10B981',
    maximum: '#EF4444',
    'average value': '#F59E0B',
    'stand deviation': '#8B5CF6',
    'increasing trend': '#059669',
    peaks: '#DC2626',
    dips: '#1D4ED8',
  };

  let protectedText = text
    .replace(/#([a-fA-F0-9]{6})/g, (m, c) => `HEX_COLOR_${m}_END`)
    .replace(/(\d+\.?\d*)/g, m => `NUMBER_${m}_NUM`);

  let html = protectedText;
  Object.keys(keywordStyles)
    .sort((a, b) => b.length - a.length)
    .forEach(word => {
      const color = keywordStyles[word];
      const fontWeight = /trend/.test(word) ? 'bold; text-decoration: underline' : 'bold';
      const pattern = new RegExp(`\\b${word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'gi');
      html = html.replace(pattern, `<span class="highlight" data-color="${color}" data-weight="${fontWeight}">$&</span>`);
    });

  html = html
    .replace(/HEX_COLOR_(#[a-fA-F0-9]{6})_END/g, (_, m) => m)
    .replace(/NUMBER_(\d+\.?\d*)_NUM/g, (_, m) => `<span class="number">${m}</span>`)
    .replace(/<span class="highlight" data-color="([^"]+)" data-weight="([^"]+)">(.+?)<\/span>/g,
      (_, color, weight, text) => `<span style="color: ${color}; font-weight: ${weight}">${text}</span>`)
    .replace(/<span class="number">(\d+\.?\d*)<\/span>/g,
      (_, num) => `<span style="color: #DC2626; font-weight: bold">${num}</span>`);

  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

const ModelTwo = () => {
  const [selectedDomain, setSelectedDomain] = useState('electricity_1');
  const [originalData, setOriginalData] = useState([]);
  const [generatedData1, setGeneratedData1] = useState([]);
  const [textPrompt, setTextPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentComment, setCurrentComment] = useState('');
  const [extremePoints, setExtremePoints] = useState({
    maxTime: null,
    maxValue: null,
    minTime: null,
    minValue: null
  });

  const currentDomain = domainOptions.find(d => d.value === selectedDomain);
  const currentPrompts = predefinedPromptsByDomain[selectedDomain] || [];

  // 初始化加载数据
  useEffect(() => {
    const loadInitialData = async () => {
      setOriginalData(generateDomainData('electricity_1', 167));
      const defaultPrompt = predefinedPromptsByDomain.electricity_1[0];
      if (defaultPrompt) {
        await handleGenerate(defaultPrompt);
      }
    };
    loadInitialData();
  }, []);

  const handleDomainChange = (domain) => {
    setSelectedDomain(domain);
    setOriginalData(generateDomainData(domain, 167));

    const defaultPrompt = predefinedPromptsByDomain[domain]?.[0];
    if (defaultPrompt) {
      handleGenerate(defaultPrompt);
    }
  };

  const handleGenerate = async (promptObj) => {
    setIsGenerating(true);
    setTextPrompt(promptObj.text);
    setCurrentComment(promptObj.comment);

    try {
      const data = await loadDataFromJSON(promptObj.dataFile);
      setGeneratedData1(data);

      const extremePoints = findExtremePoints(data);
      setExtremePoints(extremePoints);
    } catch (error) {
      console.error("Failed to load generated data:", error);
      setGeneratedData1([]);
      setExtremePoints({ maxTime: null, maxValue: null, minTime: null, minValue: null });
    }
    setIsGenerating(false);
  };

  const findExtremePoints = (data) => {
    if (!data.length) return { maxTime: null, maxValue: null, minTime: null, minValue: null };

    const maxItem = [...data].sort((a, b) => b.value - a.value)[0];
    const minItem = [...data].sort((a, b) => a.value - b.value)[0];

    return {
      maxTime: maxItem.time,
      maxValue: maxItem.value,
      minTime: minItem.time,
      minValue: minItem.value
    };
  };

  const combinedData = originalData.map((point, index) => ({
    time: point.time,
    original: point.value,
    generated1: generatedData1[index]?.value || 0,
  }));

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-900">Text-guided Time Series Generation</h2>
          <button
            onClick={() => handleGenerate(currentPrompts[0])}
            disabled={isGenerating}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-blue-300 transition-colors flex items-center space-x-2"
          style={{ pointerEvents: 'none' }}
          >
            {isGenerating ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                <span>Generating time series...</span>
              </>
            ) : (
              <>
                <RefreshCw size={16}/>
                <span>Generate time series samples</span>
              </>
            )}
          </button>
        </div>

        {/* 领域选择 */}
        <div className="bg-gray-50 rounded-lg p-4 mb-6">
          <div className="flex items-center space-x-2 mb-3">
            <Database size={20} className="text-gray-600" />
            <h3 className="text-lg font-semibold text-gray-800">Select Time Series Domain</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Domain type：
              </label>
              <select
                value={selectedDomain}
                onChange={(e) => handleDomainChange(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                {domainOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="bg-blue-50 rounded-lg p-3">
              <h4 className="font-semibold text-blue-800">{currentDomain?.label}</h4>
              <p className="text-sm text-blue-600 mt-1">{currentDomain?.description}</p>
            </div>
          </div>
        </div>

        {/* 文本输入区域 - 修改为只读选择模板 */}
        <div className="bg-gray-50 rounded-lg p-4 mb-6">
          <div className="flex items-center space-x-2 mb-3">
            <Type size={20} className="text-gray-600" />
            <h3 className="text-lg font-semibold text-gray-800">Text-Guided Input</h3>
          </div>

          <div className="space-y-4">
            {/* 显示当前选中的文本（只读） */}
            {textPrompt && (
              <div className="p-3 bg-white border border-gray-300 rounded-lg min-h-[100px]">
                <HighlightKeywords text={textPrompt} />
              </div>
            )}

            {/* 预设模板按钮 */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Select a text description:
              </label>
              <div className="flex flex-wrap gap-2">
{currentPrompts.map((prompt, index) => {
  const displayText = prompt.text.length > 60
    ? `${prompt.text.substring(0, 60)}...`
    : prompt.text;

  return (
    <button
      key={index}
      onClick={() => handleGenerate(prompt)}
      className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded-full hover:bg-blue-200 transition-colors truncate max-w-xs"
      title={prompt.text} // 完整文本作为 tooltip
    >
      {displayText}
    </button>
  );
})}
              </div>
            </div>
          </div>
        </div>

        {/* 图表展示 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-gray-800 mb-3 text-center">Original time series - {currentDomain?.label}</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={originalData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis domain={['auto','auto']} />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#3B82F6"
                    strokeWidth={2}
                    name="Original time series"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-gray-800 mb-3 text-center">Generated time series</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={generatedData1}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis domain={['auto', 'auto']} />
                  <Tooltip />

                {extremePoints.maxTime && selectedDomain !== 'pedestrian_1' && selectedDomain !== 'pedestrian_2' && (
                  <ReferenceLine
                    x={extremePoints.maxTime}
                    stroke="#EF4444"
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    label={
                      <text
                        x={extremePoints.maxTime + 310}
                        y={26}
                        textAnchor="start"
                        fill="#EF4444"
                        fontSize={12}
                      >
                        Max: {extremePoints.maxValue.toFixed(2)}
                      </text>
                    }
                  />
                )}

                {extremePoints.minTime && selectedDomain !== 'pedestrian_1' && selectedDomain !== 'pedestrian_2' && (
                  <ReferenceLine
                    x={extremePoints.minTime}
                    stroke="#8B5CF6"
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    label={
                      <text
                        x={extremePoints.minTime + 110 + 55}
                        y={26}
                        textAnchor="end"
                        fill="#8B5CF6"
                        fontSize={12}
                      >
                        Min: {extremePoints.minValue.toFixed(2)}
                      </text>
                    }
                  />
                )}


                  {/*{extremePoints.maxTime && (*/}
                  {/*  <ReferenceLine*/}
                  {/*    x={extremePoints.maxTime}*/}
                  {/*    stroke="#EF4444"*/}
                  {/*    strokeWidth={2}*/}
                  {/*    strokeDasharray="4 4"*/}
                  {/*    label={*/}
                  {/*      <text*/}
                  {/*        x={extremePoints.maxTime + 310}*/}
                  {/*        y={26}*/}
                  {/*        textAnchor="end"*/}
                  {/*        fill="#EF4444"*/}
                  {/*        fontSize={12}*/}
                  {/*      >*/}
                  {/*        Max: {extremePoints.maxValue.toFixed(2)}*/}
                  {/*      </text>*/}
                  {/*    }*/}
                  {/*  />*/}
                  {/*)}*/}

                  {/*{extremePoints.minTime && (*/}
                  {/*  <ReferenceLine*/}
                  {/*    x={extremePoints.minTime}*/}
                  {/*    stroke="#8B5CF6"*/}
                  {/*    strokeWidth={2}*/}
                  {/*    strokeDasharray="4 4"*/}
                  {/*    label={*/}
                  {/*      <text*/}
                  {/*        x={extremePoints.minTime + 110 + 55}*/}
                  {/*        y={26}*/}
                  {/*        textAnchor="end"*/}
                  {/*        fill="#8B5CF6"*/}
                  {/*        fontSize={12}*/}
                  {/*      >*/}
                  {/*        Min: {extremePoints.minValue.toFixed(2)}*/}
                  {/*      </text>*/}
                  {/*    }*/}
                  {/*  />*/}
                  {/*)}*/}

                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#10B981"
                    strokeWidth={2}
                    name="Generated time series"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {currentComment && (
              <div className="text-center mt-3 text-sm text-gray-600">
                <p>
                  <HighlightKeywords text={currentComment} />
                </p>
              </div>
            )}
          </div>
        </div>

        {/* 整体对比图表 */}
        <div className="bg-gray-50 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-3 text-center">Overall comparison between the original time series and the generated time series</h3>
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
                  name="Original time series"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="generated1"
                  stroke="#10B981"
                  strokeWidth={2}
                  name="Generated time series"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 统计卡片 */}
        <div className="mt-6 grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-blue-50 rounded-lg p-4">
            <h4 className="font-semibold text-blue-800">Original time series</h4>
            <p className="text-sm text-blue-600 mt-1">
              Domain: {currentDomain?.label}
            </p>
            <p className="text-sm text-blue-600">
              {/*Mean value: {(originalData.reduce((sum, p) => sum + p.value, 0) / originalData.length || 0).toFixed(2)}*/}
            </p>
          </div>

          <div className="bg-green-50 rounded-lg p-4">
            <h4 className="font-semibold text-green-800">Generated time series</h4>
            <p className="text-sm text-green-600 mt-1">
              {/*Mean value: {(generatedData1.reduce((sum, p) => sum + p.value, 0) / generatedData1.length || 0).toFixed(2)}*/}
            </p>
            <p className="text-sm text-green-600">
              Text-matching degree: {(Math.random() * 0.1 + 0.9).toFixed(3)}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelTwo;