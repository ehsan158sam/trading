# D:\AdvancedTradingSystem\clean_file.py
import os
from pathlib import Path

def clean_null_bytes_from_file(file_path_str: str):
    """
    Reads a file, removes all null bytes (\x00), and overwrites the file.
    It tries to preserve the original encoding if it's UTF-8, otherwise saves as UTF-8.
    """
    file_path = Path(file_path_str)
    if not file_path.exists():
        print(f"Error: File not found at '{file_path}'.")
        return False

    try:
        # خواندن فایل به صورت باینری برای پیدا کردن بایت های Null
        with open(file_path, 'rb') as f:
            content_bytes = f.read()

        if b'\x00' not in content_bytes:
            print(f"No null bytes found in '{file_path}'. File is already clean in terms of null bytes.")
            # با این حال، برای اطمینان از انکودینگ UTF-8، دوباره با UTF-8 ذخیره می کنیم
            try:
                with open(file_path, 'r', encoding='utf-8') as f_read_utf8:
                    original_content = f_read_utf8.read()
                with open(file_path, 'w', encoding='utf-8') as f_write_utf8:
                    f_write_utf8.write(original_content)
                print(f"File '{file_path}' re-saved with UTF-8 encoding.")
            except UnicodeDecodeError:
                print(f"Warning: Could not decode '{file_path}' as UTF-8. Trying to clean null bytes and save as UTF-8.")
                # اگر نتوانست به عنوان UTF-8 بخواند، بایت های Null را حذف و با UTF-8 ذخیره می کند
                cleaned_bytes_for_utf8 = content_bytes.replace(b'\x00', b'')
                with open(file_path, 'w', encoding='utf-8', errors='replace') as f_write_utf8_forced:
                    f_write_utf8_forced.write(cleaned_bytes_for_utf8.decode('utf-8', errors='replace'))
                print(f"Null bytes (if any) removed and file '{file_path}' forcefully saved as UTF-8 (invalid chars replaced).")
            except Exception as e_enc:
                print(f"Error during UTF-8 re-save for '{file_path}': {e_enc}")

            return True # حتی اگر Null byte نداشته، ممکن است مشکل انکودینگ بوده باشد

        # حذف بایت های Null
        cleaned_bytes = content_bytes.replace(b'\x00', b'')
        
        # تلاش برای نوشتن با انکودینگ UTF-8
        try:
            # ابتدا سعی می کنیم به عنوان متن utf-8 دیکود کنیم و سپس بنویسیم
            cleaned_text_utf8 = cleaned_bytes.decode('utf-8')
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text_utf8)
            print(f"Successfully removed null bytes and saved '{file_path}' with UTF-8 encoding.")
        except UnicodeDecodeError:
            # اگر دیکود به UTF-8 با خطا مواجه شد (یعنی فایل اصلی UTF-8 نبوده و کاراکترهای نامعتبر داشته)
            # فایل را به صورت باینری (پس از حذف Null) می نویسیم و هشدار می دهیم
            # یا با errors='replace' به UTF-8 تبدیل می کنیم
            print(f"Warning: Could not decode file as UTF-8 after removing null bytes. File might have non-UTF-8 characters.")
            with open(file_path, 'w', encoding='utf-8', errors='replace') as f_write_utf8_forced:
                f_write_utf8_forced.write(cleaned_bytes.decode('utf-8', errors='replace'))
            print(f"Null bytes removed and file '{file_path}' forcefully saved as UTF-8 (invalid original chars replaced).")
        except Exception as e_write:
            print(f"Error writing cleaned content to '{file_path}': {e_write}")
            # در صورت خطا، سعی می کنیم فایل اصلی را (بدون بایت Null) به صورت باینری بنویسیم
            try:
                with open(file_path, 'wb') as f_bin_write:
                    f_bin_write.write(cleaned_bytes)
                print(f"Successfully removed null bytes and saved '{file_path}' in its original (binary) encoding.")
            except Exception as e_bin_write:
                print(f"CRITICAL: Could not write back to file '{file_path}' even in binary: {e_bin_write}")
                return False
        return True

    except Exception as e:
        print(f"An unexpected error occurred while processing '{file_path}': {e}")
        return False

if __name__ == "__main__":
    # --- فایلی که می خواهید تمیز کنید را اینجا مشخص کنید ---
    # file_to_clean_path = r"D:\AdvancedTradingSystem\modules\data_nexus.py"
    file_to_clean_path = r"D:\AdvancedTradingSystem\modules\environments\trading_env.py"
    # ----------------------------------------------------

    print(f"Attempting to clean file: {file_to_clean_path}")
    if clean_null_bytes_from_file(file_to_clean_path):
        print("\nFile processing finished.")
        print("Please check the file content and encoding (should be UTF-8).")
        print("Then, try running your pytest command again.")
    else:
        print("\nFile cleaning failed or encountered issues.")