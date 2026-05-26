import matplotlib.pyplot as plt
import numpy as np

# 1. Khởi tạo dữ liệu
models = ['Full LLM', 'Full SLM', 'HERA Framework']
accuracy = [98.0, 10.0, 80.0]

# Định nghĩa màu sắc: LLM xanh đậm, SLM xanh nhạt, HERA màu cam để làm nổi bật
colors = ['#1f77b4', '#aec7e8', '#ff7f0e'] 

# 2. Tạo khung bản vẽ
fig, ax1 = plt.subplots(figsize=(9, 6))

# 3. Vẽ biểu đồ cột cho Độ chính xác (Accuracy)
bars = ax1.bar(models, accuracy, color=colors, width=0.5, edgecolor='black', alpha=0.9)

# Tối ưu hóa trục tung (Y-axis) hiển thị % Độ chính xác
ax1.set_ylabel('Độ chính xác (Accuracy %)', fontsize=12, fontweight='bold', color='#1f77b4', labelpad=10)
ax1.set_ylim(0, 115)  # Để chừa khoảng trống phía trên đầu cột cho text
ax1.tick_params(axis='y', labelcolor='#1f77b4')
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# 4. Hiển thị số liệu chính xác trên đầu mỗi cột
for bar in bars:
    height = bar.get_height()
    ax1.annotate(f'{height}%',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5),  # Đẩy chữ lên phía trên cột 5 điểm
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

# 5. Tạo một hộp thông tin (Text Box) để làm nổi bật tính năng tiết kiệm của HERA
info_text = (
    "GIẢI PHÁP HYBRID (HERA):\n"
    "• Sử dụng 26.0% SLM cho tác vụ dễ\n"
    "• Tiết kiệm 26.0% chi phí gọi LLM\n"
    "• Giữ được 80.0% độ chính xác tổng thể"
)
props = dict(boxstyle='round,pad=0.6', facecolor='#fff2cc', edgecolor='#ff7f0e', alpha=0.9)
# Đặt hộp thông tin ở vị trí trống bên phải biểu đồ
ax1.text(0.35, 0.45, info_text, transform=ax1.transAxes, fontsize=10.5,
         verticalalignment='top', bbox=props, linespacing=1.5)

# 6. Tiêu đề biểu đồ
plt.title('SO SÁNH HIỆU NĂNG VÀ KHẢ NĂNG TIẾT KIỆM CHI PHÍ CỦA HERA', 
          fontsize=13, fontweight='bold', pad=20, color='#333333')

# 7. Tối ưu bố cục và LƯU FILE KHÔNG HIỂN THỊ CỬA SỔ (Phù hợp cho Server/SSH)
plt.tight_layout()
output_filename = 'hera_chart.png'
plt.savefig(output_filename, dpi=300) # dpi=300 giúp ảnh sắc nét khi chèn vào báo cáo

print("=========================================================")
print(f"🎉 Xuất biểu đồ thành công!")
print(f"📁 File ảnh đã được lưu tại: {output_filename}")
print("=========================================================")