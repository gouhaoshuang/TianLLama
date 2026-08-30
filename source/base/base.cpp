#include "base/base.h"

#include <string>
#include <utility>

namespace base
{

Status::Status(
    int code ,
    std::string message
): code_(code) , message_(message){}


Status& Status::operator=(int code){
    code_ = code;
    return *this;
}

bool Status::operator==(int code) const{
    return code_ == code;
}
bool Status::operator!=(int code) const{
    return code_ != code;
}

Status::operator int() const{  return code_; }

Status::operator bool() const{  return code_ == kSuccess; }

int32_t  Status::get_err_code() const{ return code_; }

const std::string& Status::get_err_message() const{  return this->message_; }

void Status::set_err_message(const std::string& err_msg) {  message_ = err_msg ; }

} // namespace base
