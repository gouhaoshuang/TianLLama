#pragma once
#include <cstdint>
#include <string>

namespace base{

enum class DeviceType : uint8_t{
    kDeviceUnknown = 0,
    kDeviceCPU = 1,
    kDeviceGPU = 2,
};

enum class MemcpyKind : uint8_t {
    kMemcpyCPU2CPU = 1,
    kMemcpyCPU2GPU = 2,
    kMemcpyGPU2CPU = 3,
    kMemcpyGPU2GPU = 4,
};

enum class DataType : uint8_t {
    kDataTypeUnknown = 0,
    kDataTypeFp32 = 1,
    kDataTypeFp16 = 2,
    kDataTypeInt8 = 3,
};

enum  StatusCode : uint8_t {
    kSuccess = 0,
    kFunctionUnImplement = 1,
    kPathNotValid = 2,
    kModelParseError = 3,
    kInternalError = 4,
    kKeyValueHasExist = 5,
    kInvalidArgument = 7,
};

class Status{

public:
    Status(
        int code = StatusCode::kSuccess,
        std::string message = ""
    );

    Status(const Status& other) = default;
    Status& operator=(const Status& other) = default;

    Status& operator=(int code);
    bool operator==(int code) const;
    bool operator!=(int code) const;

    operator int() const;

    operator bool() const;

    int32_t  get_err_code() const;

    const std::string& get_err_message() const;

    void set_err_message(const std::string& message) ;


private:
    int code_ = StatusCode::kSuccess;
    std::string message_;
};


}