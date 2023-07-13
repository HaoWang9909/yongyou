import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objs as go
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

def data_process(df, time_column, part_column, sales_column):
    #仅取出时间、零件、销量三列
    df = df[[part_column, sales_column, time_column]]

    # 将时间列转换为日期格式
    df[time_column] = pd.to_datetime(df[time_column])

    # 将零件列转换为字符串格式
    df[part_column] = df[part_column].astype(str)

    # 将销量列转换为浮点数格式
    df[sales_column] = df[sales_column].astype(float)
    
    # 针对每个零件编号（'PART_NO'）进行分组
    grouped = df.groupby(part_column)

    # 创建一个新的DataFrame来存储处理后的数据
    capped_df = pd.DataFrame()

    for part_no, group in grouped:
        Q1 = group[sales_column].quantile(0.25)
        Q3 = group[sales_column].quantile(0.75)
        IQR = Q3 - Q1
        upper_limit = Q3 + 1.5 * IQR
        # 将过大值替换为上限
        group.loc[group[sales_column] > upper_limit, sales_column] = upper_limit
        # 将小于0的值替换为0
        group.loc[group[sales_column] < 0, sales_column] = 0
        # 将处理后的数据添加到capped_df
        capped_df = pd.concat([capped_df, group])
    # 重置索引
    capped_df.reset_index(drop=True, inplace=True)
    # 返回处理后的数据
    return capped_df


def main_predict(all_data,time_column,part_column,sales_column,part_to_predict,days_to_predict,predict_time):
    # 将CREATE_DATE转换为DateTime数据类型
    all_data[time_column] = pd.to_datetime(all_data[time_column])
    ## 计算每个配件在每一天的销量
    daily_sales = all_data.groupby([part_column, pd.Grouper(key=time_column, freq='D')])[sales_column].sum().reset_index()
    # 找出最早和最晚的日期
    start_date = daily_sales[time_column].min()
    end_date = daily_sales[time_column].max()
    # 创建日期范围
    all_days = pd.date_range(start_date, end_date, freq='D')
    def process_group(group):
        part_no = group[part_column].iloc[0]  # 从分组数据中获取 part_column
        group.set_index(time_column, inplace=True)
        group = group.reindex(all_days, fill_value=0)
        group[part_column] = part_no
        return group.reset_index().rename(columns={'index': time_column})
    all_data_daily = daily_sales.groupby(part_column).apply(process_group)
    # 由于 groupby().apply() 方法会创建一个多索引 DataFrame，你可能需要重置索引
    all_data_daily.reset_index(drop=True, inplace=True)
    # 获取所有零件的编号
    part_ids = all_data_daily[part_column].unique()

    # 将所有的日常数据复制到一个新的变量中
    part_data = all_data_daily
    # 选择小于等于这个日期的数据作为训练集
    df_train = part_data
    # 从训练集的 DataFrame 中创建时间序列 DataFrame，需要指定 ID 列和时间戳列
    train_data = TimeSeriesDataFrame.from_data_frame(
        df_train,
        id_column=part_column,
        timestamp_column=time_column
        )
    # 创建时间序列预测器，需要指定预测长度、目标变量名、评估指标等参数
    predictor = TimeSeriesPredictor(
            prediction_length=days_to_predict,
            target=sales_column,
            eval_metric="sMAPE",
        )

    # 在训练数据上拟合预测器，需要指定预设参数、时间限制等
    predictor.fit(
            train_data,
            presets="best_quality",
            time_limit=predict_time,
        )

    # 在训练数据上进行预测
    predictions = predictor.predict(train_data)
    return predictions
    

# 添加标题
st.markdown("<h1 style='text-align: center; color: blue;'>零件销量预测</h1>", unsafe_allow_html=True)
#添加页面说明
st.markdown("<h3 style='text-align: center; color: black;'>本页面用于预测汽车零件销量，请先在左侧上传文件</h3>", unsafe_allow_html=True)

# 在侧边栏添加文件上传器, 用户可以添加csv文件
uploaded_file = st.sidebar.file_uploader('上传文件', type=['csv'])

# 判断文件是否上传成功
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # 显示‘数据前五行如下’
    st.subheader('01 数据概览（前五行）')

    # 显示数据前五行
    st.write(df.head())

    # 将列名转换为大写
    df.columns = df.columns.str.upper()

    # 提取所有列
    all_columns = df.columns.tolist()

    # 在侧边栏显示‘请选择数据列’
    st.sidebar.subheader('请选择数据列')

    # 用户指定时间列
    time_column = st.sidebar.selectbox('请选择时间列', all_columns)

    # 用户指定零件列
    part_column = st.sidebar.selectbox('请选择零件列', all_columns)

    # 用户指定销量列
    sales_column = st.sidebar.selectbox('请选择销量列', all_columns)

    # 显示‘在侧边栏中选择列后浏览数据’
    st.subheader('02 在侧边栏中选择列后浏览数据（可以跳过）')

    if time_column != part_column != sales_column:
        # 选择要显示的零件
        parts_to_show = st.multiselect('请选择要展示的零件', options=list(df[part_column].unique()))

        # 绘制选择的零件的销量趋势图
        fig = go.Figure()

        for part in parts_to_show:
            part_df = df[df[part_column] == part]
            fig.add_trace(go.Scatter(x=part_df[time_column], y=part_df[sales_column], mode='lines', name=part))

        fig.update_layout(title='销量趋势图')
        st.plotly_chart(fig)
        st.subheader('03 开始预测')
        #弹出“请选择你要预测的零件”选择框
        part_to_predict = st.multiselect('请选择要预测的零件', options=list(df[part_column].unique()))
        #弹出“请选择你要预测的时间长度(建议不超过30天)”输入框
        days_to_predict = st.number_input('请选择你要预测的时间长度(建议不超过30天)', min_value=1, max_value=61, value=7)
        #弹出“请选择模型运行时长限制(一般而言时间越长效果越好，默认为1分钟)”输入框
        predict_time = st.number_input('请选择模型运行时长限制(单位：秒)', min_value=0, value=60)
        #弹出确认按钮
        if st.button('确认'):
            st.write('正在进行数据预处理，处理方法为使用75%分位数填补过大值')
            processed_df = data_process(df, time_column, part_column, sales_column)
            st.write('数据预处理完成')
            st.write('正在预测，请稍等')
            predictions = main_predict(processed_df, time_column,part_column,sales_column,part_to_predict,days_to_predict,predict_time)
            st.write('预测完成，预测结果如下')
            st.subheader('04 预测结果展示')
            #打印预测结果
            for item_id in part_to_predict:
                y_pred = predictions.loc[item_id]['mean'].sum()
                #打印预测结果
                st.write('零件编号为', item_id, '的零件在未来', days_to_predict, '天的销量预测为', y_pred)
            
        
