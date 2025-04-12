
import pandas as pd
import matplotlib.pyplot as plt
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
    recency_max=int(df['Recency'].max()*2)
    frequency_min=df['Frequency'].min()
    frequency_max=int(df['Frequency'].max()*2)
    monetary_min=int(df['Monetary'].min())
    monetary_max=int(df['Monetary'].max()*2)

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
    giai_thich_ClusterName(st,df_new['ClusterName'].iloc[0])


# -----------------------------------------------------------------------------------
def select_one_customers_by_id(customer_id_list,df,isRandomCus,st):
    options = ['']+customer_id_list['Member_number'].tolist()
    occupation = st.selectbox('Chọn khách hàng theo id (Member_number):',options,
        format_func=lambda x: 'Chọn một khách hàng' if x == '' else x,
    )

    # selected_cus=pd.DataFrame()
    if occupation!='':
        st.write("Khách hàng được chọn:", occupation)
        selected_cus=df[df['Member_number']==occupation]
        if not selected_cus.empty:
            if not isRandomCus:
                selected_cus=selected_cus.groupby(['ClusterName','Recency','Frequency','Monetary']).agg({'TotalPrice':'sum'})
            selected_cus.reset_index(inplace=True)
            st.markdown(format_table(selected_cus).to_html(), unsafe_allow_html=True)    
            giai_thich_ClusterName(st,selected_cus['ClusterName'].iloc[0])


# -----------------------------------------------------------------------------------
def giai_thich_ClusterName(st,cluster_name=None):
    if cluster_name=='Champions':
        st.write(
        ''' 
        **Champions: Khách hàng VIP**
        - Mô tả:
            - Recency thấp, Frequency và Monetary cao, quy mô lớn (Count cao). 
            - Đây là nhóm khách hàng có giá trị nhất, mang lại nhiều doanh thu cho doanh nghiệp.
        - Đề xuất:            
            - Ưu tiên hàng đầu: Cung cấp dịch vụ khách hàng đặc biệt, ưu tiên xử lý đơn hàng, hỗ trợ 24/7.
            - Chương trình VIP độc quyền: Tạo ra các chương trình ưu đãi, quà tặng đặc biệt chỉ dành cho nhóm VIP.
            - Tăng cường tương tác: Tổ chức sự kiện, chương trình tri ân dành riêng cho khách hàng VIP.            
        ''')
    elif cluster_name=='Lost Customers':
        st.write(
        ''' 
        **Lost Customers: Khách hàng đã mất**
        - Mô tả:
            - Recency cao, Frequency và Monetary thấp, quy mô lớn (Count cao). 
            - Nhóm này mua hàng không thường xuyên và chi tiêu ít.
        - Đề xuất:            
            - Khảo sát: Thực hiện khảo sát để hiểu lý do họ ngừng mua hàng.
            - Khuyến mãi đặc biệt: Gửi email/tin nhắn với ưu đãi đặc biệt, chương trình tri ân.
            - Cá nhân hóa: Cá nhân hóa nội dung tiếp thị dựa trên lịch sử mua hàng trước đó.            
        ''')
    elif cluster_name=='New Customers':
        st.write(            
        ''' 
        **New Customers: Khách hàng mới**
        - Mô tả:
            - Recency, Frequency và Monetary ở mức trung bình, quy mô lớn (Count cao). 
            - Đây có thể là nhóm khách hàng mới, đang tìm hiểu và thử nghiệm sản phẩm/dịch vụ. Hoặc khách hàng tiềm năng chưa quyết định gắn bó lâu dài.
        - Đề xuất:            
            - Hướng dẫn & hỗ trợ: Cung cấp hướng dẫn sử dụng sản phẩm/dịch vụ, hỗ trợ tận tình để tạo trải nghiệm tích cực.
            - Khuyến mãi mua hàng lần sau: Gửi mã giảm giá cho lần mua tiếp theo.
            - Xây dựng lòng trung thành: Thực hiện các chương trình khuyến khích mua hàng thường xuyên.
        ''')     
    elif cluster_name=='Loyal Customers':
        st.write(            
        ''' 
        **Loyal Customers: Khách hàng trung thành**
        - Mô tả:
            - Recency thấp, Frequency cao nhưng Monetary thấp, quy mô nhỏ (Count thấp). 
            - Đây là nhóm khách hàng trung thành nhưng chi tiêu ít
        - Đề xuất:            
            - Chăm sóc: Duy trì mối quan hệ tốt, gửi lời cảm ơn, quà tặng vào các dịp đặc biệt.
            - Chương trình khách hàng thân thiết: Xây dựng chương trình tích điểm, ưu đãi dành riêng cho nhóm này.
            - Tăng giá trị đơn hàng: Khuyến khích mua thêm sản phẩm/dịch vụ bằng cách giới thiệu sản phẩm bổ sung hoặc bán chéo.            
        ''')
    elif cluster_name=='Potential Customers':
        st.write(            
        ''' 
        **Potential Customers: Khách hàng tiềm năng**
        - Mô tả:
            - Recency cao, Frequency thấp nhưng Monetary cao, quy mô nhỏ (Count thấp). 
            - Đây là nhóm khách hàng đã từng chi tiêu nhiều nhưng đã lâu không mua hàng. 
            - Họ có tiềm năng trở thành khách hàng trung thành nếu được chăm sóc đúng cách.  
        - Đề xuất:
            - Gửi email marketing giới thiệu sản phẩm mới, chương trình khuyến mãi hấp dẫn.
            - Thu hút: Chạy các chiến dịch quảng cáo, khuyến mãi hấp dẫn để thu hút sự chú ý và khuyến khích mua hàng lần đầu.
            - Giới thiệu sản phẩm: Gửi email giới thiệu sản phẩm/dịch vụ mới, phù hợp với sở thích của họ (dựa trên dữ liệu đã có).
            - Tăng nhận diện: Tăng cường nhận diện thương hiệu thông qua các kênh tiếp thị khác nhau.            
        ''')    
    else:
        st.write(
        ''' 
        **Champions: Khách hàng VIP**
        - Mô tả:
            - Recency thấp, Frequency và Monetary cao, quy mô lớn (Count cao). 
            - Đây là nhóm khách hàng có giá trị nhất, mang lại nhiều doanh thu cho doanh nghiệp.
        - Đề xuất:            
            - Ưu tiên hàng đầu: Cung cấp dịch vụ khách hàng đặc biệt, ưu tiên xử lý đơn hàng, hỗ trợ 24/7.
            - Chương trình VIP độc quyền: Tạo ra các chương trình ưu đãi, quà tặng đặc biệt chỉ dành cho nhóm VIP.
            - Tăng cường tương tác: Tổ chức sự kiện, chương trình tri ân dành riêng cho khách hàng VIP.  
        
        **Lost Customers: Khách hàng đã mất**
        - Mô tả:
            - Recency cao, Frequency và Monetary thấp, quy mô lớn (Count cao). 
            - Nhóm này mua hàng không thường xuyên và chi tiêu ít.
        - Đề xuất:            
            - Khảo sát: Thực hiện khảo sát để hiểu lý do họ ngừng mua hàng.
            - Khuyến mãi đặc biệt: Gửi email/tin nhắn với ưu đãi đặc biệt, chương trình tri ân.
            - Cá nhân hóa: Cá nhân hóa nội dung tiếp thị dựa trên lịch sử mua hàng trước đó.    
        
        **New Customers: Khách hàng mới**
        - Mô tả:
            - Recency, Frequency và Monetary ở mức trung bình, quy mô lớn (Count cao). 
            - Đây có thể là nhóm khách hàng mới, đang tìm hiểu và thử nghiệm sản phẩm/dịch vụ. Hoặc khách hàng tiềm năng chưa quyết định gắn bó lâu dài.
        - Đề xuất:            
            - Hướng dẫn & hỗ trợ: Cung cấp hướng dẫn sử dụng sản phẩm/dịch vụ, hỗ trợ tận tình để tạo trải nghiệm tích cực.
            - Khuyến mãi mua hàng lần sau: Gửi mã giảm giá cho lần mua tiếp theo.
            - Xây dựng lòng trung thành: Thực hiện các chương trình khuyến khích mua hàng thường xuyên.
        
        **Loyal Customers: Khách hàng trung thành**
        - Mô tả:
            - Recency thấp, Frequency cao nhưng Monetary thấp, quy mô nhỏ (Count thấp). 
            - Đây là nhóm khách hàng trung thành nhưng chi tiêu ít
        - Đề xuất:            
            - Chăm sóc: Duy trì mối quan hệ tốt, gửi lời cảm ơn, quà tặng vào các dịp đặc biệt.
            - Chương trình khách hàng thân thiết: Xây dựng chương trình tích điểm, ưu đãi dành riêng cho nhóm này.
            - Tăng giá trị đơn hàng: Khuyến khích mua thêm sản phẩm/dịch vụ bằng cách giới thiệu sản phẩm bổ sung hoặc bán chéo.   
        
        **Potential Customers: Khách hàng tiềm năng**
        - Mô tả:
            - Recency cao, Frequency thấp nhưng Monetary cao, quy mô nhỏ (Count thấp). 
            - Đây là nhóm khách hàng đã từng chi tiêu nhiều nhưng đã lâu không mua hàng. 
            - Họ có tiềm năng trở thành khách hàng trung thành nếu được chăm sóc đúng cách.  
        - Đề xuất:
            - Gửi email marketing giới thiệu sản phẩm mới, chương trình khuyến mãi hấp dẫn.
            - Thu hút: Chạy các chiến dịch quảng cáo, khuyến mãi hấp dẫn để thu hút sự chú ý và khuyến khích mua hàng lần đầu.
            - Giới thiệu sản phẩm: Gửi email giới thiệu sản phẩm/dịch vụ mới, phù hợp với sở thích của họ (dựa trên dữ liệu đã có).
            - Tăng nhận diện: Tăng cường nhận diện thương hiệu thông qua các kênh tiếp thị khác nhau.                           
        ''')
    
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
def truc_quan_hoa_treemap(rfm_agg,modelName):    
    fig = px.treemap(
        rfm_agg,
        path=['ClusterName'],
        values='Count',
        color='ClusterName',
        hover_data=['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Percent'],
        title=f"RFM Clustering with {modelName} (tree map)"
    )

    fig.update_traces(textinfo="label+value+percent root", 
                  texttemplate='%{label}<br>%{customdata[0]} days<br>%{customdata[1]} orders<br>%{customdata[2]:.2f} $<br>%{value} customers (%{customdata[3]:.2f} %)',
                  customdata=rfm_agg[['RecencyMean', 'FrequencyMean','MonetaryMean','Percent']]
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
