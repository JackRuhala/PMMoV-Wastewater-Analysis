# Understanding the environment of Kent Countys sewers
st.title('Understanding the Environment of Kent Countys sewers')
st.write('A sewer system is not isolated from the outside world, the system experiences dramatic changes along with the environment outside the environment'
         'Below, take some time to look at how the environment of a sewer system changes with the environment outside.'
)

fig2 = px.scatter(WW_df, title = 'Kent County Sewer Water Tempature', x='Date', y='Temp', render_mode='svg')
st.plotly_chart(fig2)

# univariate graphs of grand river discharge vs flow rate vs precipitation and snow melt
fig3 = px.scatter(WW_df, title = 'Sewer Flow Rate by site', x='Date', y='FlowRate (MGD)', color ='Code', render_mode='svg')
st.plotly_chart(fig3)

# fig4 = px.scatter(WW_df_standerd, title = 'Sewer Flow Rate vs Enviroment', x='Date', y='FlowRate (MGD)')
# fig4.add_scatter(WW_df_standerd, x='Date',y='Discharge (ft^3/s)')
# st.plotly_chart(fig4)
# univariate graphs of pH of a system
fig5 = px.scatter(WW_df, title = 'Sewer Water pH by site', x='Date', y='pH', color ='Code', render_mode='svg')
st.plotly_chart(fig5)
# univariate graphs of PMMoV recorded in a system

fig5 = px.scatter(WW_df, title = 'PMMoV Gene Copys recorded in 100ml Sewer Water sample', x='Date', y='PMMoV (gc/ 100mL)', color ='Code', render_mode='svg')
st.plotly_chart(fig5)
