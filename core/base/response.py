class ResponseDTO:
    """
    通用响应数据传输对象，用于封装成功和失败的响应。

    Attributes:
        code (int): 响应状态码。
        message (str): 响应信息，通常用于描述成功或失败的原因。
        data (any): 可选的响应数据，默认值为 None。
        success (bool): 是否表示成功的标志，默认为 True。

    Methods:
        success(cls, data=None, message="Success"): 创建一个成功的响应对象。
        failed(cls, code=400, message="Failed", data=None): 创建一个失败的响应对象。
    """

    def __init__(self, code, message, data=None, success=True):
        """
            初始化响应DTO对象。

            Args:
                code (int): 响应状态码。
                message (str): 响应信息。
                data (any, optional): 响应数据，默认为 None。
                success (bool, optional): 是否成功的标志，默认为 True。
        """
        self.code = code
        self.message = message
        self.data = data
        self.success = success

    @classmethod
    def success(cls, data=None, message="Success"):
        """
        创建一个成功的响应对象。

        Args:
            data (any, optional): 成功时返回的数据，默认为 None。
            message (str, optional): 成功的响应信息，默认为 "Success"。

        Returns:
            ResponseDTO: 包含成功状态的响应对象。
        """

        return cls(code=200, message=message, data=data, success=True)

    @classmethod
    def failed(cls, code=500, message="Failed", data=None):
        """
        创建一个失败的响应对象。

        Args:
            code (int, optional): 响应状态码，默认为 400。
            message (str, optional): 失败的响应信息，默认为 "Failed"。
            data (any, optional): 可选的错误数据，默认为 None。

        Returns:
            ResponseDTO: 包含失败状态的响应对象。
        """
        return cls(code=code, message=message, data=data, success=False)
