resource "aws_instance" "ec2" {
  ami = var.ami
  instance_type = var.instance_type
  key_name =var.key_name
  tags={
      Name="aws_instance by terraform"
  }
}

resource "aws_ebs_volume" "ebs" {
  availability_zone = "${var.region}a"
  size              = var.ebs_size
  tags = {
    Name = "ebs-volume by tf"
  }
}

resource "aws_volume_attachment" "ebs_att" {
  device_name = "/dev/sdh"
  volume_id   = aws_ebs_volume.ebs.id
  instance_id = aws_instance.ec2.id
}
