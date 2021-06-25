variable "key_name" {
  type = string
}

variable "instance_type" {
  type    = string
  default = "t2.micro"
}

variable "ami" {
  type    = string
  default = "ami-010aff33ed5991201" 
}

variable "ebs_size" {
  type    = number
  default = 5
}

variable "region" {
  type    = string
  default = "ap-south-1"
}