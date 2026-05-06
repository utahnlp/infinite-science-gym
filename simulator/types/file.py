import base64
from dataclasses import dataclass
import io

import pandas as pd


@dataclass
class File:
    data: pd.DataFrame | str
    extension: str

    def __post_init__(self):
        pass

    def convert_to_extension(self) -> str:
        if self.extension == '.csv':
            return self.data.to_csv(index=False)
        elif self.extension == '.json':
            return self.data.to_json(orient='records', indent=2)
        elif self.extension == '.jsonl':
            return self.data.to_json(orient='records', lines=True)
        elif self.extension == '.xlsx':
            with io.BytesIO() as buffer:
                self.data.to_excel(buffer, index=False, engine='xlsxwriter')
                buffer.seek(0)
                excel_bytes = buffer.read()
            base64_encoded_string = base64.b64encode(excel_bytes).decode('utf-8')
            return base64_encoded_string
        elif self.extension in ['.log', '.txt']:
            return self.data.to_csv(index=False)
        elif self.extension == '.md':
            assert type(self.data) is str
            return self.data

    def convert_to_extension_base64(self) -> str:
        if self.extension == '.xlsx':
            with io.BytesIO() as buffer:
                self.data.to_excel(buffer, index=False, engine='xlsxwriter')
                buffer.seek(0)
                excel_bytes = buffer.read()
            encoded_string = base64.b64encode(excel_bytes).decode('utf-8')
        elif self.extension in ['.csv', '.log', '.txt']:
            s = self.data.to_csv(index=False)
            bytes_object = s.encode('utf-8')
            encoded_bytes = base64.b64encode(bytes_object)
            encoded_string = encoded_bytes.decode('ascii')
        elif self.extension in ['.json', '.jsonl']:
            encoded_string = self.convert_to_extension()
        return encoded_string
