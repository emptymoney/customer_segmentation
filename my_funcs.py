
import pandas as pd
import matplotlib.pyplot as plt
import squarify
import plotly.express as px

# -----------------------------------------------------------------------------------
def gan_nhan_cum_cho_khach_hang(df,model,isPredict=False):
    if isPredict:
        df["Cluster"]=model.predict(df)
    else:
        df["Cluster"] = model.labels_

    # Tạo dictionary ánh xạ giữa giá trị số và tên cụm
    cluster_mapping = {
        0: 'Potential Customers',
        1: 'Lost Customers',
        2: 'Loyal Customers',
        3: 'New Customers',
        4: 'Champions'
    }

    # Tạo cột ClusterName và ánh xạ giá trị từ cột Cluster
    df['ClusterName'] = df['Cluster'].map(cluster_mapping)
    return df


# -----------------------------------------------------------------------------------
def tinh_gia_tri_tb_RFM(df):
    # Phân tích kết quả, xem các đặc điểm của từng cụm
    df.groupby('ClusterName').agg({
        'Recency':'mean',
        'Frequency':'mean',
        'Monetary':['mean', 'count']}).round(2)

    # Calculate average values for each RFM_Level, and return a size of each segment
    rfm_agg2 = df.groupby('ClusterName').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']}).round(0)

    rfm_agg2.columns = rfm_agg2.columns.droplevel()
    rfm_agg2.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
    rfm_agg2['Percent'] = round((rfm_agg2['Count']/rfm_agg2.Count.sum())*100, 2)

    # Reset the index
    rfm_agg2 = rfm_agg2.reset_index()

    return rfm_agg2


# -----------------------------------------------------------------------------------
def get_top_products_info(group,df_merged, top_n=3):
    top_products = group['productName'].value_counts().index[:top_n]
    top_categories=group['Category'].value_counts().index[:top_n]
    # top_categories=group['TotalPrice'].value_co.index[:top_n]
    cluster_name = df_merged.loc[group.index[0], 'Cluster']

    data = {'Cluster': [cluster_name],
        f'Top{top_n}_Popular_Products': [', '.join(top_products)],
        f'Top_{top_n}_Popular_Category': [','.join(top_categories)]
    }    

    return pd.DataFrame(data)


# -----------------------------------------------------------------------------------
def get_list_customers(df):
    unique_customers = df['Member_number'].unique()
    columns = ['Member_number']
    
    return pd.DataFrame(unique_customers,columns=columns)

# -----------------------------------------------------------------------------------
def format_table(df):
    styler = df.style.set_table_styles(
        [
            {'selector': 'th', 'props': [('text-align', 'center')]},  # Canh phải tiêu đề cột
            {'selector': 'td', 'props': [('text-align', 'center')]},  # Canh giữa nội dung
            {'selector': 'th:first-child', 'props': [('background-color', 'lightblue')]},  # Nền xanh nhạt cho tiêu đề cột đầu tiên
        ]    
    )
    return styler


# -----------------------------------------------------------------------------------
def select_one_customers_by_RFM(df,model,st):
    recency_min=df['Recency'].min()
    recency_max=int(df['Recency'].max()*1.5)
    frequency_min=df['Frequency'].min()
    frequency_max=int(df['Frequency'].max()*1.5)
    monetary_min=int(df['Monetary'].min()*1.5)
    monetary_max=int(df['Monetary'].max()*1.5)

    R = st.slider("Recency", 0, recency_max, int((recency_max-recency_min)/6))
    st.write("Recency: ", R)

    F = st.slider("Frequency", 0, frequency_max, int((frequency_max-frequency_min)/6))
    st.write("Frequency: ", F)

    M = st.slider("Monetary", 0, monetary_max, int((monetary_max-monetary_min)/6))
    st.write("Monetary: ", M)

    cols=['Recency','Frequency','Monetary']
    df_new=pd.DataFrame([[R,F,M]],columns=cols)
    df_new=gan_nhan_cum_cho_khach_hang(df_new,model,isPredict=True)

    st.markdown(format_table(df_new).to_html(), unsafe_allow_html=True)


# -----------------------------------------------------------------------------------
def select_one_customers_by_id(customer_id_list,df,st):
    options = ['']+customer_id_list['Member_number'].tolist()
    occupation = st.selectbox('Chọn khách hàng theo id (Member_number):',options,
        format_func=lambda x: 'Chọn một khách hàng' if x == '' else x,
    )
    if occupation!='':
        st.write("Khách hàng được chọn:", occupation)
        selected_cus=df[df['Member_number']==occupation]
        selected_cus=selected_cus.groupby(['ClusterName','Recency','Frequency','Monetary']).agg({'TotalPrice':'sum'})
        selected_cus.reset_index(inplace=True)
        st.markdown(format_table(selected_cus).to_html(), unsafe_allow_html=True)    


# -----------------------------------------------------------------------------------
# def truc_quan_hoa_treemap2(rfm_agg,modelName):
#     fig = plt.gcf()
#     ax = fig.add_subplot()
#     fig.set_size_inches(14, 10)

#     colors_dict2 = {'Cluster0':'yellow','Cluster1':'royalblue', 'Cluster2':'cyan',
#                 'Cluster3':'red', 'Cluster4':'purple', 'Cluster5':'green', 'Cluster6':'gold'}

#     squarify.plot(sizes=rfm_agg['Count'],
#                 text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
#                 color=colors_dict2.values(),
#                 label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg.iloc[i])
#                         for i in range(0, len(rfm_agg))], alpha=0.5 )


#     plt.title(f"RFM Clustering with {modelName} (tree map)",fontsize=16,fontweight="bold")
#     plt.axis('off')
#     return fig


# -----------------------------------------------------------------------------------
def truc_quan_hoa_treemap(rfm_agg2,modelName):    
    fig = px.treemap(
        rfm_agg2,
        path=['ClusterName'],
        values='Count',
        color='ClusterName',
        hover_data=['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Percent'],
        title=f"RFM Clustering with {modelName} (tree map)"
    )
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    return fig


# -----------------------------------------------------------------------------------
def truc_quan_hoa_scatter(rfm_agg2,modelName):
    fig = px.scatter(rfm_agg2, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="ClusterName",
           hover_name="ClusterName", size_max=100,opacity=0.7)
    fig.update_layout(title=f'RFM Clustering with {modelName} (bubble chart)')
    return fig


# -----------------------------------------------------------------------------------
def truc_quan_hoa_scatter_3d_avg(rfm_agg2,modelName):
    fig = px.scatter_3d(rfm_agg2, x='RecencyMean', y='FrequencyMean', z='MonetaryMean',
                        size="FrequencyMean",
                        color='ClusterName', size_max=100, opacity=0.7)
    fig.update_layout(title=f'RFM Clustering with {modelName} (bubble chart 3d)',
                    scene=dict(xaxis_title='Recency',
                                yaxis_title='Frequency',
                                zaxis_title='Monetary'))
    return fig 


# -----------------------------------------------------------------------------------
def truc_quan_hoa_scatter_3d_data(rfm_agg2,df,modelName):
    fig = px.scatter_3d(df, x='Recency', y='Frequency', z='Monetary',
                        color='ClusterName', size_max=10, opacity=0.7)
    fig.update_layout(title=f'RFM Clustering with {modelName} (scatter plot)',
                    scene=dict(xaxis_title='Recency',
                                yaxis_title='Frequency',
                                zaxis_title='Monetary'))
    return fig     

def ve_cac_bieu_do(rfm_agg,df,st,modelName):
    fig_treemap=truc_quan_hoa_treemap(rfm_agg,modelName)
    st.write("")
    st.plotly_chart(fig_treemap)

    fig_scatter=truc_quan_hoa_scatter(rfm_agg,modelName)
    st.write("")
    st.plotly_chart(fig_scatter)

    fig_scatter_3d_avg=truc_quan_hoa_scatter_3d_avg(rfm_agg,modelName)
    st.write("")
    st.plotly_chart(fig_scatter_3d_avg)

    fig_scatter_3d_data=truc_quan_hoa_scatter_3d_data(rfm_agg,df,modelName)
    st.write("")
    st.plotly_chart(fig_scatter_3d_data)     

if __name__ == "__main__":
    pass
